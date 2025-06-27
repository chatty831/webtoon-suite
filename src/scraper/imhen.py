import requests
from bs4 import BeautifulSoup

from src.scraper.scraper_utils import fetch_all, fetch_html, get_all_images_base64


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

    return await get_all_images_base64(img_links)
    # return img_links
