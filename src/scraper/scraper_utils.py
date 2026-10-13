import asyncio
import base64
import json
import os
import sys
from io import BytesIO
from queue import Queue

import cloudscraper
import requests
from bs4 import BeautifulSoup

semaphore = asyncio.Semaphore(30)

# Path to manually saved Cloudflare cookies (fallback if cloudscraper alone fails)
COOKIES_FILE = os.path.join(os.path.dirname(__file__), "cf_cookies.json")

_scraper_session = None


def get_scraper_session():
    """
    Get or create a shared cloudscraper session.
    Automatically handles Cloudflare JS challenges.
    Falls back to manually saved cookies from cf_cookies.json if present.
    """
    global _scraper_session
    if _scraper_session is not None:
        return _scraper_session

    _scraper_session = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )

    # Load manually saved cookies if they exist (fallback for tough challenges)
    if os.path.exists(COOKIES_FILE):
        try:
            with open(COOKIES_FILE, "r") as f:
                cookies = json.load(f)
            for cookie in cookies:
                if cookie["name"] == "_user_agent":
                    # Apply saved user-agent to match the browser that solved the challenge
                    _scraper_session.headers["User-Agent"] = cookie["value"]
                else:
                    _scraper_session.cookies.set(
                        cookie["name"], cookie["value"], domain=cookie.get("domain", "")
                    )
            print(f"[scraper] Loaded cookies from {COOKIES_FILE}")
        except Exception as e:
            print(f"[scraper] Warning: Failed to load cookies from {COOKIES_FILE}: {e}")

    return _scraper_session


def reset_scraper_session():
    """Reset the shared session (e.g. after updating cookies)."""
    global _scraper_session
    _scraper_session = None


async def fetch_html(url):
    """
    Fetch the HTML page source and text from a given URL using cloudscraper.
    Automatically bypasses Cloudflare challenges when possible.

    Args:
        url (str): The URL to fetch.

    Returns:
        tuple: A tuple containing the HTML page source and text.
    """
    session = get_scraper_session()
    async with semaphore:
        response = await asyncio.to_thread(session.get, url)
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


async def fetch_image_base64(url, headers=None):
    """
    Fetch an image from a URL and return its base64 encoded string.

    Args:
        url (str): The URL of the image to fetch.
        headers (dict, optional): Optional HTTP headers (e.g. Referer).

    Returns:
        str: The base64 encoded string of the image, or None if fetching fails.
    """
    session = get_scraper_session()
    for attempt in range(3):
        try:
            async with semaphore:
                response = await asyncio.to_thread(session.get, url, headers=headers)
                if response.status_code == 200:
                    buffered = BytesIO(response.content)
                    return base64.b64encode(buffered.getvalue()).decode("utf-8")
                else:
                    print(f"Attempt {attempt + 1}: Failed to fetch {url} with status code {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error fetching {url}: {e}")
    return None


async def get_all_images_base64(image_urls, headers=None):
    """
    Fetch images from a list of URLs asynchronously and return their base64 encoded strings.

    Args:
        image_urls (list): A list of image URLs to fetch.
        headers (dict, optional): Optional HTTP headers to include in the requests.

    Returns:
        list: A list of base64 encoded strings of the images, or None for failed fetches.
    """
    tasks = [asyncio.create_task(fetch_image_base64(url, headers=headers)) for url in image_urls]
    base64_images = await asyncio.gather(*tasks)
    return base64_images
