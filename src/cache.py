from typing import Any, Dict, List, Optional, Union

import diskcache as dc
from cachetools import TTLCache, cached

from src.logger import logger

image_ttl_cache = TTLCache(maxsize=10, ttl=3600)

translate_dc_cache = dc.Cache(
    "translate_cache_dir",
    size_limit=10_000_000_000,  # 10GB cache size limit
    cull_limit=0,  # Don't cull entries automatically
    eviction_policy="least-recently-used",
)


def update_cache_and_results(
    processed_images: List[str], process_indices: List[int], cache_keys: List[str], final_results: List[str]
) -> List[str]:
    """Update cache with new results and merge with existing cached results."""
    for i, processed_img in enumerate(processed_images):
        orig_idx = process_indices[i]

        # Update the results array
        try:
            final_results[orig_idx] = processed_img
        except IndexError:
            continue

        # Cache the result using parameter-specific key
        cache_key = cache_keys[orig_idx]
        translate_dc_cache[cache_key] = processed_img
        logger.info(f"Cached result for key {cache_key[:15]}...")

    return final_results
