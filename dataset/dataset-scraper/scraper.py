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

init(autoreset=True)

current_dir = os.path.dirname(os.path.abspath(__file__))


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
                    + f"\u23f3 Connection issue or rate limited. Retrying in {wait:.2f}s... ({retry + 1}/{max_retries})"
                )
                time.sleep(wait)
                retry += 1
            else:
                print(Fore.RED + f"❌ Unhandled error: {str(e)}")
                raise e
    print(Fore.RED + f"🚫 Max retries exceeded for URL: {url}")
    return None


class HouseData:
    def __init__(
        self,
        title,
        price,
        bedroom,
        bathroom,
        carport,
        lt,
        lb,
        badges,
        agent,
        updated,
        location,
        link,
        description,
    ):
        self.title = title
        self.price = price
        self.bedroom = bedroom
        self.bathroom = bathroom
        self.carport = carport
        self.lt = lt
        self.lb = lb
        self.badges = badges
        self.agent = agent
        self.updated = updated
        self.location = location
        self.link = link
        self.description = description

    def printObject(self):
        print(
            f"""
                Title: {self.title}
                Price: {self.price}
                Bedroom: {self.bedroom}
                Bathroom: {self.bathroom}
                Carport: {self.carport}
                LT: {self.lt}
                LB: {self.lb}
                Badges: {self.badges}
                Agent: {self.agent}
                Updated: {self.updated}
                Location: {self.location}
                Link: {self.link}
                Description: {self.description}
            """
        )


def scrapeweb(start_page, end_page):
    listOfHouse = []
    ua_path = os.path.join(current_dir, "ua.txt")
    user_agents = load_user_agents(ua_path)

    for page in range(start_page, end_page + 1):
        try:
            base_url_rumah123 = f"https://www.rumah123.com/jual/daerah-istimewa-yogyakarta/rumah/?page={page}"
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

            property_list = soup_page.find_all("div", class_="card-featured")

            if not property_list:
                print(Fore.YELLOW + "⚠️ No more properties found. Stopping.")
                break

            for prop in tqdm(
                property_list, desc=f"🔍 Parsing properties on page {page}"
            ):
                try:
                    title_elem = prop.find("h2")
                    title = title_elem.get_text(strip=True) if title_elem else "N/A"

                    price_elem = prop.find(
                        "div", class_="card-featured__middle-section__price"
                    )
                    price = price_elem.get_text(strip=True) if price_elem else "N/A"

                    bedroom = bathroom = carport = "N/A"
                    attribute_section = prop.find(
                        "div", class_="card-featured__middle-section__attribute"
                    )
                    if attribute_section:
                        attributes = attribute_section.find_all(
                            "div", class_="ui-molecules-list__item"
                        )
                        for idx, attr in enumerate(attributes):
                            text = attr.get_text(strip=True)
                            if idx == 0:
                                bedroom = text
                            elif idx == 1:
                                bathroom = text
                            elif idx == 2:
                                carport = text

                    lt = lb = "N/A"
                    attribute_infos = prop.find_all("div", class_="attribute-info")
                    if len(attribute_infos) >= 1:
                        lt_span = attribute_infos[0].find("span")
                        if lt_span:
                            lt = lt_span.get_text(strip=True)
                    if len(attribute_infos) >= 2:
                        lb_span = attribute_infos[1].find("span")
                        if lb_span:
                            lb = lb_span.get_text(strip=True)

                    badges = []
                    badge_elems = prop.find_all("a", class_="quick-label-badge")
                    for badge in badge_elems:
                        span = badge.find("span")
                        if span:
                            badge_text = span.get_text(strip=True)
                            if badge_text and badge_text not in badges:
                                badges.append(badge_text)

                    header_badge_section = prop.find(
                        "div", class_="card-featured__middle-section__header-badge"
                    )
                    if header_badge_section:
                        header_badges = header_badge_section.find_all(
                            "div", attrs={"data-test-id": "badge-depth"}
                        )
                        for badge in header_badges:
                            badge_text = badge.get_text(strip=True)
                            if badge_text and badge_text not in badges:
                                badges.append(badge_text)

                    agent_elem = prop.find("p", class_="name")
                    agent = agent_elem.get_text(strip=True) if agent_elem else "N/A"

                    updated_elem = prop.find(
                        "p", string=lambda text: text and "Diperbarui" in text
                    )
                    updated = (
                        updated_elem.get_text(strip=True) if updated_elem else "N/A"
                    )

                    location = "N/A"
                    h2_tag = prop.find("h2")
                    if h2_tag:
                        next_span = h2_tag.find_next("span")
                        if next_span:
                            location = next_span.get_text(strip=True)

                    link = "N/A"
                    link_elem = prop.find(
                        "a", href=lambda href: href and "/properti/" in href
                    )
                    if link_elem and "href" in link_elem.attrs:
                        link = link_elem["href"]

                    desc_elem = prop.find("p", string=lambda s: s and len(s) > 10)
                    description = desc_elem.get_text(strip=True) if desc_elem else "N/A"

                    house = HouseData(
                        title,
                        price,
                        bedroom,
                        bathroom,
                        carport,
                        lt,
                        lb,
                        badges,
                        agent,
                        updated,
                        location,
                        link,
                        description,
                    )
                    listOfHouse.append(house)
                except Exception as e:
                    print(Fore.RED + f"❌ Error parsing property: {str(e)}")
                    continue
        except Exception as e:
            print(Fore.RED + f"\n🚫 Error fetching page {page}: {str(e)}\n")
            break
    return listOfHouse


def main():
    print(
        Fore.CYAN
        + Style.BRIGHT
        + """
==============================
🏠 Rumah123 Web Scraper 🕷️
==============================
"""
    )

    listOfHouse = []
    scrapeweb(listOfHouse)

    print(Fore.GREEN + f"\n✅ Found {len(listOfHouse)} properties in total!")

    csv_path = os.path.join(current_dir, "../houses.csv")
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
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for house in listOfHouse:
            writer.writerow(
                {
                    "title": f"{house.title}",
                    "price": f"{house.price}",
                    "bedroom": f"{house.bedroom}",
                    "bathroom": f'{house.bathroom}"',
                    "carport": f'"{house.carport}"',
                    "LT": f"{house.lt}",
                    "LB": f"{house.lb}",
                    "badges": f'"{", ".join(house.badges)}',
                    "agent": f"{house.agent}",
                    "updated": f"{house.updated}",
                    "location": f"{house.location}",
                    "link": f"{house.link}",
                }
            )

    print(Fore.BLUE + f"💾 Saved {len(listOfHouse)} entries to {csv_path}")

    if listOfHouse:
        print(Fore.LIGHTYELLOW_EX + "\n🎯 Sample property:\n")
        listOfHouse[0].printObject()


if __name__ == "__main__":
    main()
