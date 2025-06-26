
from src.ml_models import UPSCALE_MODEL
import torch

def upscale_image(img, outscale=2):
    """
    Upscale an image using RealESRGAN.
    
    Args:
        img: A cv2 image (numpy array)
        model_path: Path to the model weights
        outscale: The output scale (default: 4)
        
    Returns:
        Enhanced image
    """
    with torch.no_grad():
        output, _ = UPSCALE_MODEL.enhance(img, outscale=outscale)
    return output