from typing import Any, Dict, List, Optional, Union

DEFAULT_CONFIG: Dict[str, Any] = {
    "detection_model_path": "models/bbox_models/best_640_ep_30.pt",
    "denoiser_model_path": "models/denoise_models",
    "denoise_sigma": 50,  # Default sigma for denoising
    # "denoise_sigma": 25,
    "colorizer_model_path": "models/colorize_models/henxxx.pt",
    # "colorizer_model_path": "models/colorize_models/general_purpose.zip",
    "image_tile_size": 1280,
    # "image_tile_size": 768,
    "upscale_model_path": "models/upscale_models/RealESRGAN_x4plus.pth",
    "font_model_path": "models/font_models/Wild-Words-Font-2/CC Wild Words Roman.ttf",
}
