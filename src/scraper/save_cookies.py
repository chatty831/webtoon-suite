"""
Cookie extraction helper for Cloudflare-protected sites.

When cloudscraper alone can't bypass the Cloudflare challenge, you can
solve it once in your browser and then copy the cookies here.

HOW TO USE:
-----------
1. Open the target site (e.g. https://imhentai.xxx) in Chrome/Edge/Firefox.
2. Solve the Cloudflare challenge (wait for the page to load fully).
3. Open DevTools (F12) -> Application tab -> Cookies -> select the site domain.
4. Find these cookies and copy their values:
     - cf_clearance  (required - this is the main bypass cookie)
     - __cf_bm       (optional - helps but expires quickly)
5. Run this script:
     python -m src.scraper.save_cookies
6. Paste the cookie values when prompted.

The cookies will be saved to src/scraper/cf_cookies.json and automatically
loaded by the scraper on the next request.

NOTE: cf_clearance cookies typically last ~30 minutes to a few hours depending
on the site's Cloudflare configuration. You'll need to redo this when they expire.
"""

import json
import os

COOKIES_FILE = os.path.join(os.path.dirname(__file__), "cf_cookies.json")


def save_cookies():
    print("=" * 50)
    print("  Cloudflare Cookie Saver")
    print("=" * 50)
    print()
    print("Open the target site in your browser, solve the")
    print("Cloudflare challenge, then copy cookies from DevTools.")
    print()
    print("DevTools (F12) -> Application -> Cookies -> [site]")
    print()

    domain = input("Enter the site domain (e.g. imhentai.xxx): ").strip()
    if not domain:
        print("No domain provided, exiting.")
        return

    cookies = []

    # cf_clearance is the main cookie
    cf_clearance = input("Paste cf_clearance cookie value (required): ").strip()
    if cf_clearance:
        cookies.append({
            "name": "cf_clearance",
            "value": cf_clearance,
            "domain": f".{domain}" if not domain.startswith(".") else domain,
        })
    else:
        print("cf_clearance is required! Exiting.")
        return

    # __cf_bm is optional but helpful
    cf_bm = input("Paste __cf_bm cookie value (optional, press Enter to skip): ").strip()
    if cf_bm:
        cookies.append({
            "name": "__cf_bm",
            "value": cf_bm,
            "domain": f".{domain}" if not domain.startswith(".") else domain,
        })

    # Optional: user-agent must match the browser that solved the challenge
    user_agent = input(
        "Paste your browser's User-Agent (optional, press Enter to skip):\n"
        "  (Find it at: DevTools -> Console -> navigator.userAgent)\n> "
    ).strip()
    if user_agent:
        cookies.append({
            "name": "_user_agent",
            "value": user_agent,
            "domain": domain,
        })

    with open(COOKIES_FILE, "w") as f:
        json.dump(cookies, f, indent=2)

    print()
    print(f"Saved {len(cookies)} cookie(s) to {COOKIES_FILE}")
    print("The scraper will automatically load these on the next request.")


if __name__ == "__main__":
    save_cookies()
