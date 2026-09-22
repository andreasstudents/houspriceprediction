"""
Test standalone untuk memverifikasi selector scraping rumah123.com versi baru.

Website sekarang memakai Next.js App Router dengan kartu server-rendered:
  article[data-name="ldp-listing-card"]  (dulu: div.card-featured)
Field dipetakan lewat atribut data-test-id yang stabil, bukan class CSS
Tailwind yang mudah berubah. Logika parsing ada di r123_parser.py agar
sinkron dengan scraper.py.

Jalankan:
    python test_scraper.py
Hasil:
    - debug_page1.html / debug_page2.html  (arsip HTML mentah)
    - test_output.csv                      (hasil parsing, tidak menyentuh houses.csv)
"""

import csv
import os
import random
import time

from bs4 import BeautifulSoup as soup

# Reuse fetch logic dari scraper.py (random UA + retry) tanpa duplikasi
from scraper import get_with_random_ua, load_user_agents
from r123_parser import LISTING_CARD_SELECTOR, parse_card
from colorama import Fore, Style, init

init(autoreset=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://www.rumah123.com/jual/jawa-tengah/rumah/?page={page}"


def scrape_page(page, user_agents, save_debug=True):
    """Fetch satu halaman hasil pencarian, kembalikan list dict properti."""
    url = BASE_URL.format(page=page)
    time.sleep(random.uniform(1.0, 3.0))
    print(Fore.CYAN + f"\n🚀 Fetching page {page}: {url}")

    html_page = get_with_random_ua(url, user_agents)
    if html_page is None:
        print(Fore.LIGHTRED_EX + f"⚠️ Page {page}: gagal fetch setelah retry.")
        return []

    if save_debug:
        debug_path = os.path.join(current_dir, f"debug_page{page}.html")
        with open(debug_path, "wb") as f:
            f.write(html_page)
        print(f"  💾 HTML disimpan: {debug_path} ({len(html_page):,} bytes)")

    soup_page = soup(html_page, "html.parser")
    cards = soup_page.find_all("article", attrs=LISTING_CARD_SELECTOR)
    print(f"  📦 Kartu ditemukan: {len(cards)}")

    results = []
    for i, card in enumerate(cards):
        try:
            row = parse_card(card)
            results.append(row)
        except Exception as e:
            print(Fore.RED + f"  ❌ Error parsing card {i}: {e}")
    return results


def validate(rows, page):
    """Cek kualitas ekstraksi: hitung field yang masih 'N/A'."""
    required = ["title", "price", "bedroom", "bathroom", "LT", "LB", "location", "link"]
    if not rows:
        print(Fore.RED + f"  ❌ Page {page}: 0 baris — selector rusak atau halaman kosong.")
        return False
    problems = 0
    for col in required:
        na = sum(1 for r in rows if r[col] == "N/A")
        if na:
            print(Fore.YELLOW + f"  ⚠️ Page {page} | kolom {col}: {na}/{len(rows)} N/A")
            problems += na
    if not problems:
        print(Fore.GREEN + f"  ✅ Page {page}: {len(rows)} baris, semua kolom wajib terisi.")
    return problems == 0


def main():
    print(Fore.CYAN + Style.BRIGHT + """
==========================================
🧪 Test Scraper rumah123.com (struktur baru)
==========================================
""")
    user_agents = load_user_agents(os.path.join(current_dir, "ua.txt"))

    all_rows = []
    all_ok = True
    for page in (1, 2):
        rows = scrape_page(page, user_agents)
        all_ok &= validate(rows, page)
        all_rows.extend(rows)

    csv_path = os.path.join(current_dir, "test_output.csv")
    headers = ["title", "price", "bedroom", "bathroom", "carport", "LT", "LB",
               "badges", "agent", "updated", "location", "link", "description"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)
    print(Fore.BLUE + f"\n💾 {len(all_rows)} baris disimpan ke {csv_path}")

    if all_rows:
        print(Fore.LIGHTYELLOW_EX + "\n🎯 Sample baris pertama:")
        for k, v in all_rows[0].items():
            print(f"  {k}: {v}")

    print(Fore.GREEN + Style.BRIGHT +
          f"\n{'✅ TEST LULUS' if all_ok and all_rows else '❌ TEST GAGAL — periksa debug_page*.html'}")
    return 0 if all_ok and all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
