import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# === Folder paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  

CSV_PATH = os.path.join(DATA_DIR, "bbc_amharic_dataset_300+.csv")

# === BBC categories with topic URLs ===
categories = {
    "Politics": "https://www.bbc.com/amharic/topics/cg7265pj1jvt",
    "Health": "https://www.bbc.com/amharic/topics/cxnyk76p0q2t",
    "Technology": "https://www.bbc.com/amharic/topics/c06gq8wx467t",
    "Sport": "https://www.bbc.com/amharic/topics/cdr56g2x71dt"
}

headers = {"User-Agent": "Mozilla/5.0"}
data = []

page_limit = 10  # Increase this to collect more

# === Start scraping ===
for label, base_url in categories.items():
    print(f"\n Scraping category: {label}")
    collected = 0

    for page in range(1, page_limit + 1):
        url = f"{base_url}?page={page}"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            articles = soup.find_all("a", href=True)

            for article in articles:
                title = article.get_text().strip()
                link = article["href"]

                if len(title) > 25 and "/amharic/articles/" in link:
                    if not link.startswith("http"):
                        link = "https://www.bbc.com" + link

                    data.append({
                        "text": title,
                        "label": label,
                        "url": link
                    })
                    collected += 1

            print(f"   Page {page}: {collected} collected so far.")
            time.sleep(1)

        except Exception as e:
            print(f" Error on page {page} of {label}: {e}")
            continue

# === Save to CSV ===
print("\n Scraping complete. Total articles collected:", len(data))
df = pd.DataFrame(data)
df.drop_duplicates(subset="text", inplace=True)
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f" Saved to: {CSV_PATH}")
