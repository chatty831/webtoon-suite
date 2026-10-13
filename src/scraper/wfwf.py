import urllib.parse

from bs4 import BeautifulSoup

from src.scraper.scraper_utils import fetch_html, get_all_images_base64


async def wfwf_scraper(page_link):
    # page link of type https://wfwf515.com/view?toon=73686&num=111
    res_text, res_html = await fetch_html(page_link)
    soup = BeautifulSoup(res_html, "html.parser")
    domain = urllib.parse.urlparse(page_link).netloc

    div = soup.find(id="vimg-area")
    imgs = div.find_all("img")

    img_urls = []
    for img in imgs:
        actual_src = img.get("data-src") or img.get("src")
        if not actual_src:
            continue
        if actual_src.startswith("http"):
            img_urls.append(actual_src)
        else:
            img_urls.append("https://" + domain + actual_src)

    base64_images = await get_all_images_base64(img_urls)
    return base64_images
