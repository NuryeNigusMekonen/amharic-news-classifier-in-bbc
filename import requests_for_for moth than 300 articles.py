import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

#  BBC categories with topic URLs
categories = {
    "Politics": "https://www.bbc.com/amharic/topics/cg7265pj1jvt",
    "Health": "https://www.bbc.com/amharic/topics/cxnyk76p0q2t",
    "Technology": "https://www.bbc.com/amharic/topics/c06gq8wx467t",
    "Sport": "https://www.bbc.com/amharic/topics/cdr56g2x71dt"
}

headers = {"User-Agent": "Mozilla/5.0"}
data = []

# we can increase page number as we want depending on the data set amount we require
page_limit = 5

# Loop through each category and multiple pages
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
            print(f"  Error on page {page} of {label}: {e}")
            continue

print("\n Scraping complete. Total articles collected:", len(data))

#  Save to CSV
df = pd.DataFrame(data)
df.drop_duplicates(subset="text", inplace=True)
df.to_csv("bbc_amharic_dataset_300+.csv", index=False, encoding="utf-8-sig")
print(" Saved as 'bbc_amharic_dataset_300+.csv'")
