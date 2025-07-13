import asyncio
import base64
from io import BytesIO
from queue import Queue

import requests
from bs4 import BeautifulSoup

from src.scraper.scraper_utils import fetch_all, fetch_html, get_all_images_base64

semaphore = asyncio.Semaphore(30)

# Assuming semaphore is defined elsewhere in your code
# semaphore = asyncio.Semaphore(10)  # Example semaphore


async def fetch_image_base64_imhen(url, _attempted_extensions=None):
    """
    Fetch an image from a URL and return its base64 encoded string.

    Args:
        url (str): The URL of the image to fetch.
        _attempted_extensions (set): Internal parameter to track attempted extensions.

    Returns:
        str: The base64 encoded string of the image, or None if fetching fails.
    """
    if _attempted_extensions is None:
        _attempted_extensions = set()

    # Extract current extension
    current_ext = None
    if url.endswith(".webp"):
        current_ext = ".webp"
    elif url.endswith(".jpg"):
        current_ext = ".jpg"
    elif url.endswith(".jpeg"):
        current_ext = ".jpeg"
    elif url.endswith(".png"):
        current_ext = ".png"

    # Add current extension to attempted set
    if current_ext:
        _attempted_extensions.add(current_ext)

    for attempt in range(3):
        try:
            async with semaphore:
                response = await asyncio.to_thread(requests.get, url, timeout=30)
                if response.status_code == 200:
                    buffered = BytesIO(response.content)
                    return base64.b64encode(buffered.getvalue()).decode("utf-8")
                else:
                    print(f"Attempt {attempt + 1}: Failed to fetch {url} with status code {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error fetching {url}: {e}")

    # If the original link fails, try with different extensions
    extensions_to_try = [".jpg", ".jpeg", ".png", ".webp"]

    for ext in extensions_to_try:
        if ext not in _attempted_extensions:
            if current_ext:
                new_url = url.replace(current_ext, ext)
            else:
                new_url = url + ext

            print(f"Trying alternative extension: {new_url}")
            result = await fetch_image_base64_imhen(new_url, _attempted_extensions)
            if result is not None:
                return result

    return None


async def get_all_images_base64_imhen(image_urls):
    """
    Fetch images from a list of URLs asynchronously and return their base64 encoded strings.

    Args:
        image_urls (list): A list of image URLs to fetch.

    Returns:
        list: A list of base64 encoded strings of the images, or None for failed fetches.
    """
    tasks = [asyncio.create_task(fetch_image_base64_imhen(url)) for url in image_urls]
    base64_images = await asyncio.gather(*tasks)
    return base64_images


async def imhen_scraper(page_link):
    # page link of type https://imhentai.xxx/gallery/1477517/
    res_text, res_html = await fetch_html(page_link)
    soup = BeautifulSoup(res_html, "html.parser")
    pages = int("".join(filter(str.isdigit, soup.find(class_="pages").text)))

    # we convert to https://imhentai.xxx/view/1477517/1/
    view_link = page_link.replace("/gallery", "/view") + "1/"
    res_text, res_html = await fetch_html(view_link)
    soup = BeautifulSoup(res_html, "html.parser")
    img_link = soup.find(id="gimg").get("src")

    # img link is of type https://m10.imhentai.xxx/029/jl8wnitme3/1.webp
    img_links = []
    base_link = img_link.split("/1")[0] + "/"
    extension = img_link.split("/1")[-1]
    for i in range(pages):
        img_links.append(f"{base_link}{i+1}{extension}")

    base64_images = await get_all_images_base64_imhen(img_links)
    return base64_images
