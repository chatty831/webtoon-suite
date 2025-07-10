import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from sahi.models.ultralytics import UltralyticsDetectionModel

from src.colorize.colorizer import initialize_colorizator
from src.colorize.denoise.denoiser import FFDNetDenoiser
from src.constants import DEVICE
from src.enhance.realesrgan import RealESRGANer
from src.data.config import DEFAULT_CONFIG

YOLO_MODEL_PATH = DEFAULT_CONFIG.get("detection_model_path")
DENOISER_MODEL_PATH = DEFAULT_CONFIG.get("denoiser_model_path")
COLORIZER_MODEL_PATH = DEFAULT_CONFIG.get("colorizer_model_path")
UPSCALE_MODEL_PATH = DEFAULT_CONFIG.get("upscale_model_path")
FONT_MODEL_PATH = DEFAULT_CONFIG.get("font_model_path")

# Initialize the detection model
YOLO_MODEL = UltralyticsDetectionModel(
    model_path=YOLO_MODEL_PATH,
    device="cuda",
    confidence_threshold=0.5,
    image_size=640,
)

DENOISER = FFDNetDenoiser("cuda", weights_dir=DENOISER_MODEL_PATH)

COLORIZER = initialize_colorizator(
    # generator_path="models/colorize_models/generator_1_rgb_2.pt",
    generator_path=COLORIZER_MODEL_PATH,
).to(device=DEVICE, dtype=torch.float32) 

UPSCALE_MODEL = RealESRGANer(
    scale=4,
    model_path=UPSCALE_MODEL_PATH,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=1024,
    half=True,  # Use half precision
)
