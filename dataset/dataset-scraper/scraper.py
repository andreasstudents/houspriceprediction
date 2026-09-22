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
from telegram_notifier import TelegramNotifier

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


def scrapeweb(
    start_page,
    end_page,
    notifier=None,
    notify_every=0,
    delay_min=1.0,
    delay_max=3.0,
):
    """Fetch halaman SRP Jawa Tengah dan parse setiap kartu listing.

    - notify_every > 0: kirim notif progres Telegram tiap kelipatan N halaman.
    - Loop berhenti sendiri saat halaman tanpa kartu listing (habis).
    """
    listOfHouse = []
    ua_path = os.path.join(current_dir, "ua.txt")
    user_agents = load_user_agents(ua_path)

    last_page_done = start_page - 1
    for page in range(start_page, end_page + 1):
        try:
            base_url_rumah123 = BASE_URL.format(page=page)
            time.sleep(random.uniform(delay_min, delay_max))
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

            last_page_done = page
            if notifier and notify_every > 0 and page % notify_every == 0:
                notifier.notify(
                    f"📄 Progress: page {page} selesai\n"
                    f"✅ {len(listOfHouse):,} listing terkumpul"
                )
        except Exception as e:
            print(Fore.RED + f"\n🚫 Error fetching page {page}: {str(e)}\n")
            break
    return listOfHouse, last_page_done


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
    parser.add_argument(
        "--notify-every",
        type=int,
        default=500,
        help="Kirim notif Telegram tiap N halaman (0 = matikan notif progres)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=1.0,
        help="Delay minimum antar halaman dalam detik (default 1.0)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=3.0,
        help="Delay maksimum antar halaman dalam detik (default 3.0)",
    )
    parser.add_argument(
        "--tg-token",
        default=None,
        help="Token bot Telegram (fallback: env TG_BOT_TOKEN). Kosong = tanpa notifikasi.",
    )
    parser.add_argument(
        "--tg-chat",
        default=None,
        help="Chat ID tujuan Telegram (fallback: env TG_CHAT_ID)",
    )
    args = parser.parse_args()

    notifier = TelegramNotifier(args.tg_token, args.tg_chat)
    if notifier.enabled:
        print(Fore.GREEN + "📣 Notifikasi Telegram: AKTIF")
    else:
        print(Fore.YELLOW + "📣 Notifikasi Telegram: nonaktif (tanpa token/chat id)")

    print(
        Fore.CYAN
        + Style.BRIGHT
        + """
==============================
🏠 Rumah123 Web Scraper 🕷️
==============================
"""
    )

    if notifier.enabled:
        notifier.notify(
            f"🚀 Scraping dimulai\n"
            f"Halaman {args.start_page} s/d "
            f"{args.end_page if args.end_page < 9999 else 'habis (auto-stop)'}\n"
            f"Output: {os.path.basename(args.output)}"
        )

    start_time = time.time()
    listOfHouse, last_page_done = scrapeweb(
        args.start_page,
        args.end_page,
        notifier=notifier,
        notify_every=args.notify_every,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
    )

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

    # Ringkasan akhir + kirim CSV via Telegram, lalu proses selesai sendiri.
    duration_min = (time.time() - start_time) / 60
    if notifier.enabled:
        summary = (
            f"✅ Scraping SELESAI\n"
            f"📦 {len(listOfHouse):,} listing | halaman {args.start_page}–{last_page_done}\n"
            f"⏱️ Durasi: {duration_min:.1f} menit\n"
            f"💾 {os.path.basename(csv_path)}"
        )
        if listOfHouse:
            file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            if file_size_mb > 49:  # batas Bot API 50 MB
                notifier.notify(
                    summary + f"\n⚠️ File {file_size_mb:.1f} MB > 50 MB (batas Telegram), "
                    f"ambil manual via scp: {csv_path}"
                )
            else:
                sent = notifier.send_file(
                    csv_path,
                    caption=summary,
                )
                if not sent:
                    notifier.notify(
                        summary + f"\n⚠️ Gagal kirim file. File tetap ada di server: {csv_path}"
                    )
        else:
            notifier.notify(summary + "\n⚠️ Tidak ada data — cek log di VPS.")

    print(
        Fore.GREEN
        + Style.BRIGHT
        + f"\n🏁 Selesai dalam {duration_min:.1f} menit. Proses berhenti sendiri."
    )


if __name__ == "__main__":
    main()
