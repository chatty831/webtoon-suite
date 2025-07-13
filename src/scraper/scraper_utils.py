import asyncio
import base64
import os
import sys
from io import BytesIO
from queue import Queue

import requests
from bs4 import BeautifulSoup

semaphore = asyncio.Semaphore(30)


async def fetch_html(url):
    """
    Fetch the HTML page source and text from a given URL using the requests library.

    Args:
        url (str): The URL to fetch.

    Returns:
        tuple: A tuple containing the HTML page source and text.
    """
    async with semaphore:
        response: requests.Response = await asyncio.to_thread(requests.get, url)
        return response.text, response.content


async def fetch_all(urls):
    """
    Fetch HTML content from a list of URLs asynchronously with a semaphore limit of 30.

    Args:
        urls (list): A list of URLs to fetch.

    Returns:
        list: A list of tuples containing the HTML page source and text for each URL.
    """
    tasks = [asyncio.create_task(fetch_html(url)) for url in urls]
    results = await asyncio.gather(*tasks)
    return results


async def fetch_image_base64(url):
    """
    Fetch an image from a URL and return its base64 encoded string.

    Args:
        url (str): The URL of the image to fetch.

    Returns:
        str: The base64 encoded string of the image, or None if fetching fails.
    """
    for attempt in range(3):
        try:
            async with semaphore:
                response = await asyncio.to_thread(requests.get, url)
                if response.status_code == 200:
                    buffered = BytesIO(response.content)
                    return base64.b64encode(buffered.getvalue()).decode("utf-8")
                else:
                    print(f"Attempt {attempt + 1}: Failed to fetch {url} with status code {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error fetching {url}: {e}")
    return None


async def get_all_images_base64(image_urls):
    """
    Fetch images from a list of URLs asynchronously and return their base64 encoded strings.

    Args:
        image_urls (list): A list of image URLs to fetch.

    Returns:
        list: A list of base64 encoded strings of the images, or None for failed fetches.
    """
    tasks = [asyncio.create_task(fetch_image_base64(url)) for url in image_urls]
    base64_images = await asyncio.gather(*tasks)
    return base64_images
