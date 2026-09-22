import argparse
import csv
import os
import random
import time
import socket

from tqdm import tqdm
from bs4 import BeautifulSoup as soup
from colorama import Fore, Style, init
from urllib.error import URLError, HTTPError
from urllib.request import urlopen as uReq, Request

from r123_parser import LISTING_CARD_SELECTOR, parse_card

init(autoreset=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://www.rumah123.com/jual/jawa-tengah/rumah/?page={page}"


def load_user_agents(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def get_with_random_ua(url, user_agents, max_retries=5):
    retry = 0
    while retry < max_retries:
        try:
            user_agent = random.choice(user_agents)
            headers = {"User-Agent": user_agent}
            req = Request(url, headers=headers)
            with uReq(req) as response:
                return response.read()
        except (HTTPError, URLError, ConnectionRefusedError, socket.error) as e:
            error_str = str(e).lower()
            if (
                "429" in error_str
                or isinstance(e, socket.error)
                or "connection refused" in error_str
            ):
                wait = (1.2**retry) + random.uniform(0, 1)
                print(
                    Fore.MAGENTA
                    + f"⏳ Connection issue or rate limited. Retrying in {wait:.2f}s... ({retry + 1}/{max_retries})"
                )
                time.sleep(wait)
                retry += 1
            else:
                print(Fore.RED + f"❌ Unhandled error: {str(e)}")
                raise e
    print(Fore.RED + f"🚫 Max retries exceeded for URL: {url}")
    return None


def scrapeweb(start_page, end_page):
    """Fetch halaman SRP Jawa Tengah dan parse setiap kartu listing."""
    listOfHouse = []
    ua_path = os.path.join(current_dir, "ua.txt")
    user_agents = load_user_agents(ua_path)

    for page in range(start_page, end_page + 1):
        try:
            base_url_rumah123 = BASE_URL.format(page=page)
            time.sleep(random.uniform(1.0, 3.0))
            print(Fore.CYAN + f"\n\U0001f680 Scraping page {page}: {base_url_rumah123}")
            html_page = get_with_random_ua(base_url_rumah123, user_agents)

            if html_page is None:
                print(
                    Fore.LIGHTRED_EX
                    + f"⚠️ Skipping page {page} due to repeated failures."
                )
                continue

            soup_page = soup(html_page, "html.parser")

            # Kartu listing versi baru: article[data-name="ldp-listing-card"]
            property_list = soup_page.find_all("article", attrs=LISTING_CARD_SELECTOR)

            if not property_list:
                print(Fore.YELLOW + "⚠️ No more properties found. Stopping.")
                break

            for prop in tqdm(
                property_list, desc=f"🔍 Parsing properties on page {page}"
            ):
                try:
                    listOfHouse.append(parse_card(prop))
                except Exception as e:
                    print(Fore.RED + f"❌ Error parsing property: {str(e)}")
                    continue
        except Exception as e:
            print(Fore.RED + f"\n🚫 Error fetching page {page}: {str(e)}\n")
            break
    return listOfHouse


def main():
    parser = argparse.ArgumentParser(
        description="Scraper listing rumah123.com Jawa Tengah (struktur Next.js)"
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument(
        "--end-page",
        type=int,
        default=2,
        help="Halaman terakhir. Untuk scrape sampai habis, pakai angka besar (mis. 9999) — "
        "loop berhenti sendiri saat halaman kosong.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(current_dir, "../houses_jawa_tengah.csv"),
        help="Path CSV output (default terpisah dari houses.csv training Yogyakarta)",
    )
    args = parser.parse_args()

    print(
        Fore.CYAN
        + Style.BRIGHT
        + """
==============================
🏠 Rumah123 Web Scraper 🕷️
==============================
"""
    )

    listOfHouse = scrapeweb(args.start_page, args.end_page)

    print(Fore.GREEN + f"\n✅ Found {len(listOfHouse)} properties in total!")

    csv_path = args.output
    csv_headers = [
        "title",
        "price",
        "bedroom",
        "bathroom",
        "carport",
        "LT",
        "LB",
        "badges",
        "agent",
        "updated",
        "location",
        "link",
        "description",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(listOfHouse)

    print(Fore.BLUE + f"💾 Saved {len(listOfHouse)} entries to {csv_path}")

    if listOfHouse:
        print(Fore.LIGHTYELLOW_EX + "\n🎯 Sample property:\n")
        for k, v in listOfHouse[0].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
