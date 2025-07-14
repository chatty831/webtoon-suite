from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from src.constants import DEVICE
from src.logger import logger
from src.ml_models import COLORIZER


async def colorize_model_change(model_path: str):
    """
    Endpoint to change the colorization model.
    This is a placeholder for future implementation.
    """
    if not model_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model path is required")
    try:
        # Placeholder for changing the colorization model
        COLORIZER.load_weights(model_path, DEVICE)
        return JSONResponse(content={"current_colorization_model": model_path}, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error changing colorization model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error changing colorization model"
        )
