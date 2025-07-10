import requests
from bs4 import BeautifulSoup

from src.scraper.scraper_utils import fetch_all, fetch_html, get_all_images_base64

async def one_punch_scraper(page_link):
    # link of the type https://onepunchmanmangaa.com/one-punch-man-manga-chapter-120/
    res_text, res_html = await fetch_html(page_link)
    soup = BeautifulSoup(res_html, "html.parser")
    
    big_fig = soup.find("figure", class_="wp-block-gallery has-nested-images columns-1 is-cropped wp-block-gallery-1 is-layout-flex wp-block-gallery-is-layout-flex")
    img_els = big_fig.find_all("img")
    img_links = [img_el.get("src") for img_el in img_els]
    img_links = [link for link in img_links if "data:image/svg+xml" not in link]  # Filter out base64 images
    return await get_all_images_base64(img_links)
    # return img_links