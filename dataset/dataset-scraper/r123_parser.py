"""
Parser untuk struktur kartu listing rumah123.com versi Next.js App Router.

Situs kini me-render kartu di server sebagai:
    article[data-name="ldp-listing-card"]
dengan field yang bisa diandalkan lewat atribut data-test-id:
    srp-card-listing-title, srp-card-listing-price-main,
    srp-card-listing-location, srp-card-listing-description,
    srp-card-agent-name, srp-card-listing-title-link, srp-card-last-update

Catatan struktur atribut (div.propertyCardAttributesList), span anak berurutan:
    0-2 : span ikon (kamar tidur, kamar mandi, carport — carport bisa absen)
    sisanya: span berlabel 'LT:' / 'LB:' dengan nilai '298 m²'
"""

LISTING_CARD_SELECTOR = {"data-name": "ldp-listing-card"}


def parse_attribute_list(card):
    """Ekstrak (bedroom, bathroom, carport, lt, lb) dari satu kartu."""
    bedroom = bathroom = carport = lt = lb = "N/A"
    attrs_root = card.select_one(".propertyCardAttributesList")
    if not attrs_root:
        return bedroom, bathroom, carport, lt, lb

    icon_index = 0
    for span in attrs_root.find_all("span", recursive=False):
        label = span.find("span", class_="text-greyText")
        if label is not None:
            # LT/LB: buang label ("LT : 298 m²" -> "298 m²")
            full = span.get_text(" ", strip=True)
            value = full.split(":", 1)[1].strip() if ":" in full else full
            label_text = label.get_text(strip=True).rstrip(":").strip().upper()
            if label_text == "LT":
                lt = value
            elif label_text == "LB":
                lb = value
        else:
            text = span.get_text(strip=True)
            if icon_index == 0:
                bedroom = text
            elif icon_index == 1:
                bathroom = text
            elif icon_index == 2:
                carport = text
            icon_index += 1
    return bedroom, bathroom, carport, lt, lb


def collect_badges(card):
    """Kumpulkan badge kartu (tipe properti, booster, turun harga, hemat)."""
    badges = []
    test_ids = [
        "srp-card-property-type-badge",
        "srp-card-booster-type-badge",
        "badge-turun-harga",
        "badge-hemat",
    ]
    for tid in test_ids:
        el = card.find(attrs={"data-test-id": tid})
        if el:
            text = el.get_text(" ", strip=True)
            if text and text not in badges:
                badges.append(text)
    return badges


def parse_card(card):
    """Satu article[data-name=ldp-listing-card] -> dict satu baris listing."""
    title_el = card.find("h2", attrs={"data-test-id": "srp-card-listing-title"})
    price_el = card.find("span", attrs={"data-test-id": "srp-card-listing-price-main"})
    loc_el = card.find("p", attrs={"data-test-id": "srp-card-listing-location"})
    desc_el = card.find("p", attrs={"data-test-id": "srp-card-listing-description"})
    agent_el = card.find("p", attrs={"data-test-id": "srp-card-agent-name"})
    link_el = card.find("a", attrs={"data-test-id": "srp-card-listing-title-link"})
    upd_el = card.find("p", attrs={"data-test-id": "srp-card-last-update"})

    bedroom, bathroom, carport, lt, lb = parse_attribute_list(card)

    updated = upd_el.get_text(strip=True) if upd_el else "N/A"
    if updated.endswith("oleh"):
        updated = updated[: -len("oleh")].strip()

    return {
        "title": title_el.get_text(strip=True) if title_el else "N/A",
        "price": price_el.get_text(strip=True) if price_el else "N/A",
        "bedroom": bedroom,
        "bathroom": bathroom,
        "carport": carport,
        "LT": lt,
        "LB": lb,
        "badges": ", ".join(collect_badges(card)),
        "agent": agent_el.get_text(strip=True) if agent_el else "N/A",
        "updated": updated,
        "location": loc_el.get_text(strip=True) if loc_el else "N/A",
        "link": link_el["href"] if link_el else "N/A",
        "description": desc_el.get_text(strip=True) if desc_el else "N/A",
    }
