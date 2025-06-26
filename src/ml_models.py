from src.colorize.denoise.denoiser import FFDNetDenoiser
from sahi.models.ultralytics import UltralyticsDetectionModel
import torch
import torchvision
from src.colorize.colorizer import initialize_colorizator
from src.constants import DEVICE
from basicsr.archs.rrdbnet_arch import RRDBNet
from src.enhance.realesrgan import RealESRGANer


# Initialize the detection model
YOLO_MODEL = UltralyticsDetectionModel(
    model_path="models/bbox_models/best_640_ep_30.pt",
    device="cuda",
    confidence_threshold=0.5,
    image_size=640,
)

DENOISER = FFDNetDenoiser("cuda", weights_dir="models/denoise_models")

COLORIZER = initialize_colorizator(
    generator_path="models/colorize_models/generator_1_rgb_2.pt",
).to(device=DEVICE, dtype=torch.float32)

UPSCALE_MODEL = RealESRGANer(
    scale=4,
    model_path='models/upscale_models/RealESRGAN_x4plus.pth',
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=1024,
    half=True  # Use half precision
)