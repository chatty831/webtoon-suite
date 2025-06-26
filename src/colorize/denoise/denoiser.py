import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from src.colorize.denoise.denoise_model import FFDNet


def variable_to_cv2_image(varim: Variable):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.0).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res * 255.0).clip(0, 255).astype(np.uint8)
    else:
        raise Exception("Number of color channels not supported")
    return res


def normalize(data):
    return np.float32(data / 255.0)


class FFDNetDenoiser:
    def __init__(self, _device, _sigma=25, weights_dir="denoising/", _in_ch=3):
        self.sigma = _sigma / 255
        self.weights_dir = weights_dir
        self.channels = _in_ch
        self.device = _device

        self.model = FFDNet(num_input_channels=_in_ch)
        self.load_weights()
        self.model.eval()

    def load_weights(self):
        weights_name = "net_rgb.pth" if self.channels == 3 else "net_gray.pth"
        weights_path = os.path.join(self.weights_dir, weights_name)
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model = self.model.to(device=torch.device(self.device), dtype = torch.float32)
        self.model.load_state_dict(state_dict)
    def get_denoised_image(self, imorig, sigma=None):
        """
        Args
        ----
        imorig :  • NumPy RGB image  (H×W×3, BGR order from cv2.imread)
                  • torch tensor 3×H×W
                  • torch tensor B×3×H×W
        sigma  :  noise level in the [0,255] domain. If None, self.sigma is used.

        Returns
        -------
        Same type / shape as the input:
            - NumPy  uint8 H×W×3   when NumPy was given
            - Tensor 3×H×W         when tensor 3×H×W was given
            - Tensor B×3×H×W       when tensor B×3×H×W was given
        """

        # --------------------------------------------------
        # 0) remember what the caller gave us
        # --------------------------------------------------
        input_is_tensor = torch.is_tensor(imorig)
        if input_is_tensor:
            orig_device = imorig.device
            orig_ndim = imorig.dim()  # 3 or 4
        else:
            orig_dtype = imorig.dtype  # np.uint8 / float32 / …
            orig_order_hw = imorig.shape[:2]  # H,W for later sanity

        # --------------------------------------------------
        # 1) noise level σ  (scale to [0,1])
        # --------------------------------------------------
        cur_sigma = (sigma / 255.) if sigma is not None else self.sigma

        # --------------------------------------------------
        # 2) convert the input into a float tensor B×3×H×W in [0,1]
        # --------------------------------------------------
        if input_is_tensor:  # ---- TENSOR PATH
            tensor = imorig
            if tensor.dim() == 3:  # 3×H×W → 1×3×H×W
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() != 4:
                raise ValueError("Tensor input must be (3,H,W) or (B,3,H,W)")

            if tensor.size(1) == 1:  # gray → RGB
                tensor = tensor.repeat(1, 3, 1, 1)

            tensor = tensor.float()
            if tensor.max() > 1.2:
                tensor = tensor / 255.
        else:  # ---- NUMPY PATH
            if len(imorig.shape) < 3 or imorig.shape[2] == 1:
                imorig = np.repeat(np.expand_dims(imorig, 2), 3, 2)

            imorig = imorig[..., :3]  # drop alpha if present
            imorig = imorig.transpose(2, 0, 1)  # HWC → CHW

            if imorig.max() > 1.2:
                imorig = normalize(imorig)  # existing helper

            imorig = np.expand_dims(imorig, 0)  # add batch dim
            tensor = torch.tensor(imorig, dtype=torch.float32).to('cuda')

        # --------------------------------------------------
        # 3) pad odd spatial sizes
        # --------------------------------------------------
        expanded_h = expanded_w = False
        _, _, h, w = tensor.shape
        if h % 2 == 1:
            expanded_h = True
            tensor = torch.cat([tensor, tensor[:, :, -1:, :]], dim=2)  # duplicate last row
        if w % 2 == 1:
            expanded_w = True
            tensor = torch.cat([tensor, tensor[:, :, :, -1:]], dim=3)  # duplicate last col

        # --------------------------------------------------
        # 4) run the model using tiling inference with tile size 576
        # --------------------------------------------------
        tile_size = 576
        overlap_pixels = 0
        stride = tile_size - overlap_pixels
        B, C, H, W = tensor.shape

        # Prepare tensors for aggregation of the denoised output and a weight count for overlapping regions
        output_tensor = torch.zeros_like(tensor, device='cuda')
        count_tensor = torch.zeros((B, 1, H, W), dtype=tensor.dtype, device='cuda')

        tiles = []
        tile_info = []  # list of tuples: (b, start_h, end_h, start_w, end_w, tile_h, tile_w)

        for b in range(B):
            n_tiles_h = max(1, (H + stride - 1) // stride)
            n_tiles_w = max(1, (W + stride - 1) // stride)
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    start_h = i * stride
                    start_w = j * stride
                    if start_h + tile_size > H:
                        start_h = max(0, H - tile_size)
                    if start_w + tile_size > W:
                        start_w = max(0, W - tile_size)
                    end_h = min(start_h + tile_size, H)
                    end_w = min(start_w + tile_size, W)
                    tile = tensor[b:b+1, :, start_h:end_h, start_w:end_w]
                    tile_h = tile.shape[2]
                    tile_w = tile.shape[3]
                    pad_h = tile_size - tile_h
                    pad_w = tile_size - tile_w
                    if pad_h > 0 or pad_w > 0:
                        tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h))
                    tiles.append(tile)
                    tile_info.append((b, start_h, end_h, start_w, end_w, tile_h, tile_w))

        if tiles:
            tiles_batch = torch.cat(tiles, dim=0)  # Shape: (N, C, tile_size, tile_size)
        else:
            tiles_batch = tensor  # fallback (should not occur)

        # Move tiles to the model's device
        tiles_batch = tiles_batch.to('cuda')
        with torch.no_grad():
            N_tiles = tiles_batch.size(0)
            nsigma = torch.full((N_tiles,), cur_sigma, dtype=tiles_batch.dtype, device='cuda')
            noise_est = self.model(tiles_batch, nsigma)
            denoised_tiles = torch.clamp(tiles_batch - noise_est, 0.0, 1.0)

        # Reassemble the denoised tiles into the full image, averaging overlap if needed
        for idx, info in enumerate(tile_info):
            b, start_h, end_h, start_w, end_w, tile_h, tile_w = info
            tile_result = denoised_tiles[idx, :, :tile_h, :tile_w]
            output_tensor[b, :, start_h:end_h, start_w:end_w] += tile_result
            count_tensor[b, :, start_h:end_h, start_w:end_w] += 1

        output_tensor = output_tensor / count_tensor

        # --------------------------------------------------
        # 5) remove padding (if any)
        # --------------------------------------------------
        if expanded_h:
            output_tensor = output_tensor[:, :, :-1, :]
        if expanded_w:
            output_tensor = output_tensor[:, :, :, :-1]

        # --------------------------------------------------
        # 6) return – keep caller’s preferred format
        # --------------------------------------------------
        if input_is_tensor:
            output_tensor = output_tensor.to(orig_device)
            if orig_ndim == 3:  # caller provided 3×H×W, remove batch dim
                output_tensor = output_tensor.squeeze(0)
            return output_tensor
        else:
            return variable_to_cv2_image(output_tensor.cpu())
