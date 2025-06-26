import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from colorize.colorize_model import Generator
from src.colorize.denoise.denoiser import FFDNetDenoiser


def initialize_colorizator(generator_path):
    generator = Generator()
    gen_st_dict = torch.load(generator_path, map_location="cpu")
    generator.load_state_dict(gen_st_dict)
    return generator


denoiser = FFDNetDenoiser("cuda", weights_dir="denoising")


def colorize_batch(colorizer, images, device="cuda", dtype=torch.float32):
    """
    Colorizes a batch of RGB images using tiled inference.

    Parameters:
    - colorizer: The colorizer model
    - images: List of RGB images as numpy arrays
    - device: Device to run inference on
    - dtype: Data type for the model

    Returns:
    - List of colorized RGB images
    """
    if not images:
        return []

    results = []
    tile_size = 1280

    for img in images:
        # Ensure the image is in the correct format
        if img.dtype != np.uint8:
            raise ValueError(f"Image must be of dtype np.uint8, got {img.dtype}")

        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Image must be 3-channel RGB, got shape {img.shape}")

        # Convert RGB to grayscale
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = denoiser.get_denoised_image(bgr_img, sigma=50)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height, width = img_gray.shape

        # Convert the grayscale image to a torch tensor, normalize to [0,1], and move to GPU
        tensor_gray = torch.from_numpy(img_gray).float().to(device) / 255.0
        tensor_gray = tensor_gray.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

        # Concatenate with zeros to make shape [1, 5, H, W]
        zeros_tensor = torch.zeros(1, 4, height, width, device=device)
        tensor_gray = torch.cat((tensor_gray, zeros_tensor), dim=1)

        # Remove the batch dimension for tiling
        img_tensor = tensor_gray[0]  # shape: [5, H, W]

        # Tiling parameters
        overlap_pixels = 0
        stride = tile_size - overlap_pixels
        n_tiles_h = max(1, (height + stride - 1) // stride)
        n_tiles_w = max(1, (width + stride - 1) // stride)

        tiles = []
        positions = []

        # Loop over the grid to extract tiles
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                start_h = i * stride
                start_w = j * stride
                if start_h + tile_size > height:
                    start_h = max(0, height - tile_size)
                if start_w + tile_size > width:
                    start_w = max(0, width - tile_size)
                end_h = min(start_h + tile_size, height)
                end_w = min(start_w + tile_size, width)

                # Extract the tile
                tile = img_tensor[:, start_h:end_h, start_w:end_w]

                # Record the valid dimensions of the tile
                valid_h = tile.shape[1]
                valid_w = tile.shape[2]

                # Pad if necessary
                pad_h = tile_size - valid_h
                pad_w = tile_size - valid_w
                if pad_h > 0 or pad_w > 0:
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h))
                tiles.append(tile)
                positions.append((start_h, start_w, valid_h, valid_w))

        # Run inference on each tile individually and collect the outputs
        output_tiles_list = []
        with torch.no_grad():
            for tile in tiles:
                tile = tile.unsqueeze(0).to(device)  # Add batch dimension
                output_tile1, output_tile2 = colorizer(tile)
                output_tiles_list.append(output_tile1.squeeze(0))  # Remove batch dimension

        # Stack the output tiles to form a batch tensor
        output_tiles = torch.stack(output_tiles_list, dim=0).to(device)

        # Create an accumulator for the output image and a weight map
        output_img = torch.zeros(3, height, width, device=device, dtype=output_tiles.dtype)
        weight_map = torch.zeros(1, height, width, device=device, dtype=output_tiles.dtype)

        tile_index = 0
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                start_h, start_w, valid_h, valid_w = positions[tile_index]
                tile_out = output_tiles[tile_index]  # shape: [3, tile_size, tile_size]
                tile_valid = tile_out[:, :valid_h, :valid_w]

                output_img[:, start_h : start_h + valid_h, start_w : start_w + valid_w] += tile_valid
                weight_map[:, start_h : start_h + valid_h, start_w : start_w + valid_w] += 1.0
                tile_index += 1

        output_img /= weight_map

        # Convert to numpy with values in [0, 1]
        output_img = output_img.clamp(0, 1)
        output_np = output_img.detach().cpu().numpy()
        output_np = np.transpose(output_np, (1, 2, 0))

        # Clip the result to ensure values are within [0, 1] range and convert to uint8
        result_final = (np.clip(output_np, 0, 1) * 255).astype(np.uint8)

        results.append(result_final)

    return results
