#the following code will give news, label , url in tabular form by scraping news from the bbc website
# Scrapes from BBC Amharic
#faches data from different sections like:
   # "Politics": "https://www.bbc.com/amharic/topics/cg7265pj1jvt",
   # "Health": "https://www.bbc.com/amharic/topics/cxnyk76p0q2t",
   # "Technology": "https://www.bbc.com/amharic/topics/c06gq8wx467t",
   #"Sport": "https://www.bbc.com/amharic/topics/cdr56g2x71dt"
#Automatically assigns labels (e.g., "Politics", "Health", ....)
#Saves to CSV for machine learning use """


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
categories = {
    "Politics": "https://www.bbc.com/amharic/topics/cg7265pj1jvt",
    "Health": "https://www.bbc.com/amharic/topics/cxnyk76p0q2t",
    "Technology": "https://www.bbc.com/amharic/topics/c06gq8wx467t",
    "Sport": "https://www.bbc.com/amharic/topics/cdr56g2x71dt"
}

headers = {"User-Agent": "Mozilla/5.0"}
data = []

for label, url in categories.items():
    print(f"Scraping category: {label}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # now we wil grab articles 
        articles = soup.find_all("a", href=True)

        for article in articles:
            title = article.get_text().strip()
            link = article["href"]
            # filter only amharic news or articles 
            if len(title) > 20 and "/amharic/articles/" in link:
                if not link.startswith("http"):
                    link = "https://www.bbc.com" + link

                print(f"  -> {title}")
                data.append({
                    "text": title,
                    "label": label,
                    "url": link
                })

        time.sleep(2)  # pause between categories

    except Exception as e:
        print(f" Error scraping {label}: {e}")

# it will save to csv file in our current folder 
df = pd.DataFrame(data)
df.drop_duplicates(subset="text", inplace=True)
df.to_csv("bbc_amharic_dataset.csv", index=False, encoding="utf-8-sig")

print(f"\n Done! Collected {len(df)} headlines.")
print(" Saved as: bbc_amharic_dataset.csv")
