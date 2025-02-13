import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json
import os
import re
from urllib.parse import urljoin
import nest_asyncio
import argparse

nest_asyncio.apply()

parser =argparse.ArgumentParser()
parser.add_argument("--site", type = str, required = True,
                    help = "Site to scrape, e.g. comune.gorgonzola.mi.it")
args = parser.parse_args()

# Funzione per recuperare il contenuto di una pagina ed eseguire il Javascript
async def get_clean_text_from_html(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url)
        await page.wait_for_load_state('networkidle')  # Attendere il caricamento

        # Estrazione del codice HTML
        page_source = await page.content()
        await browser.close()

        soup = BeautifulSoup(page_source, 'html.parser')
        clean_text = soup.get_text(separator=" ", strip=True)

        return soup, clean_text

# Funzione per imporre una qualche logica di filtering sulle pagine estratte (opzionale)
def filter_link(l, f):
    if f in l:
        return True
    return False

# Funzione per estrarre i link da una pagina web
def extract_links(soup, base_url, limit=0):
    links = []

    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        absolute_link = urljoin(base_url, link)

        if filter_link(absolute_link, base_url):
            links.append(absolute_link)

        if limit and len(links) >= limit:
            break

    return links

# Funzione per salvare i dati in formato JSON nella cartella "raw_data"
def save_to_json(data, filename):
    with open("raw_data/" + filename + ".json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

async def main():
    base_url = "https://" + args.site

    soup, clean_text = await get_clean_text_from_html(base_url)

    text_tags = ''

    if not os.path.exists("raw_data"):
        os.makedirs("raw_data")

    if soup and clean_text:
        data = {
            "webpage": {
                base_url: clean_text
            },
            "category_tag": {
                base_url: text_tags
            },
            "topic_tags": {
                base_url: text_tags
            }
        }

        links = extract_links(soup, base_url, limit=0)

        print(f"\nVisiting the following {len(links)} links:")

        # Remove duplicates
        links = list(set(links))

        for idx, link in enumerate(links):
            print(f"{idx + 1}. {link}")
            # Avoid mail links
            if not (link.startswith("https:") or link.startswith(args.site) or link.startswith("www.")):
                continue
            linked_soup, linked_clean_text = await get_clean_text_from_html(link)
            category_text_tags = []
            topic_text_tags = []

            # Troviamo gli elementi categoria per espressioni regolari
            cat_element = linked_soup.find(text=re.compile(r"Categori[ae]:?(?! )"))
            if cat_element:
                child_elements = cat_element.find_all_next(attrs={"data-element": "service-topic"}) #class_="chip chip-simple chip-primary chip-lg"
                untagged_child_elements = cat_element.find_all_next(class_ = "chip chip-simple chip-primary")
                category_table = child_elements + untagged_child_elements
            else:
                category_table = []

            for element in category_table:
               category_text_tags.append(element.get_text(strip=True))

            # De-duplicazione
            category_text_tags = list(set(category_text_tags))

            # Troviamo gli elementi argomento per espressioni regolari
            if cat_element:
                arg_element = cat_element.find_next(text=re.compile(r"Argomenti:?(?! )"))
            else:
                arg_element = linked_soup.find(text = re.compile(r"Argomenti:?(?! )"))

            if arg_element:
                child_elements = arg_element.find_all_next(attrs={"data-element": "service-topic"}) #class_="chip chip-simple chip-primary chip-lg"
                untagged_child_elements = arg_element.find_all_next(class_ = "chip chip-simple chip-primary")
                topic_table = child_elements + untagged_child_elements
            else:
                topic_table = []

            for element in topic_table:
               topic_text_tags.append(element.get_text(strip=True))

            # De-duplicazione
            topic_text_tags = list(set(topic_text_tags))

            # Rimuovere eventuali tag categoria che sono stati inseriti anche nei tag argomento
            category_text_tags = list(filter(lambda s: s not in topic_text_tags, category_text_tags))

            print("Categorie:", category_text_tags)
            print("Argomenti:", topic_text_tags)

            if linked_soup and linked_clean_text:
                data["webpage"][link] = linked_clean_text
                data["category_tag"][link] = ";".join(category_text_tags)
                data["topic_tags"][link] = ";".join(topic_text_tags)

        save_to_json(data, args.site.split(".")[1])
        print(f"\nContent from the main page and the linked pages saved")
    else:
        print("Failed to retrieve the main page or no content to save.")

if __name__ == "__main__":
    asyncio.run(main())
