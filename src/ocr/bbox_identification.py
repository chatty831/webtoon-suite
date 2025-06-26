from typing import Any, Dict, List, Optional, Union

import numpy as np
from sahi.predict import get_sliced_prediction

from src.image_utils import postprocess_bboxes
from src.logger import logger
from src.ml_models import YOLO_MODEL


def detect_text_regions(images: List[np.ndarray]) -> List[List]:
    """
    Detect text regions in images using YOLO model.
    Returns processed bounding boxes for each image.
    """
    # Run object detection on all images
    results = [
        get_sliced_prediction(
            image=image,
            detection_model=YOLO_MODEL,
            slice_height=720,
            slice_width=720,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        for image in images
    ]

    # Process detection results
    processed_results = []
    for result in results:
        bboxes = []
        for detection in result.object_prediction_list:
            bbox_xywh = detection.bbox.to_xywh()
            bboxes.append(bbox_xywh)

        # Post-process bounding boxes
        final_bboxes = postprocess_bboxes(bboxes)
        processed_results.append(final_bboxes)

    return processed_results
