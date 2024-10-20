# scraping.py
# Bu dosya web scraping işlemlerini içerir.
# Örneğin, SikayetVar'dan veri çekmek için BeautifulSoup veya Scrapy kullanılabilir.

import time
import csv
from selenium import webdriver
from bs4 import BeautifulSoup
import os

# Selenium setup
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Run in headless mode (no browser window)
driver = webdriver.Chrome(options=options)

# CSV file setup
csv_file = 'data/raw/raw_data.csv'
fields = [
    'text',
    'entity',
    'sentiment',
    'konu',
    'severity',
    'bilet',
    'musteri_hizmetleri',
    'odeme',
    'uygulama',
    'passolig',
    'passolig kart',
    'diger',
    'aksiyon'
]

def scrape_page(page_url, keyword_filter=None):
    driver.get(page_url)
    time.sleep(3)  # Wait for page to load

    # Get the page source and parse it with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find all the complaint articles
    articles = soup.find_all('article', class_='card-v2 ga-v ga-c')

    complaints = []

    for article in articles:
        # Extract complaint description
        description_tag = article.find('p', class_='complaint-description')
        if description_tag:
            description = description_tag.get_text(strip=True).replace('...', '')  # Clean up ellipsis
            if keyword_filter is None or keyword_filter in description:  # Apply filter if provided
                complaints.append({
                    'text': description,
                    'entity': '',  # Placeholder for future data
                    'sentiment': '',  # Placeholder for future sentiment analysis
                    'konu': '',  # Placeholder for future data
                    'severity': '',  # Placeholder for future data
                    'bilet': '',  # Placeholder for future data
                    'musteri_hizmetleri': '',  # Placeholder for future data
                    'odeme': '',  # Placeholder for future data
                    'uygulama': '',  # Placeholder for future data
                    'passolig': '',  # Placeholder for future data
                    'passolig kart': '',  # Placeholder for future data
                    'diger': '',  # Placeholder for future data
                    'aksiyon': ''  # Placeholder for future data
                })
    
    return complaints

def scrape_all_pages(base_url, start_page=1, end_page=100, keyword_filter=None):
    all_complaints = []

    for page_num in range(start_page, end_page + 1):
        page_url = base_url + str(page_num)
        print(f"Scraping page {page_num}: {page_url}")

        complaints = scrape_page(page_url, keyword_filter)
        
        if not complaints:
            break  # Stop if no complaints found (reached the end)
        
        all_complaints.extend(complaints)

    # Save to CSV
    if os.path.exists(csv_file):
        # Read existing rows
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            existing_texts = set(row['text'] for row in rows)

        # Filter new complaints that are not in the existing CSV
        new_complaints = [complaint for complaint in all_complaints if complaint['text'] not in existing_texts]

        # Write new rows
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not rows:  # If the file was empty, write the header
                writer.writeheader()
            writer.writerows(new_complaints)  # Append new complaints

    else:
        # Create new CSV file and write header
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_complaints)

    print(f"Scraping complete. Total complaints: {len(all_complaints)}")

if __name__ == "__main__":
    # Scrape Passo complaints
    passo_base_url = 'https://www.sikayetvar.com/passo?page='
    scrape_all_pages(passo_base_url, start_page=1, end_page=70, keyword_filter='Passo')

    # Scrape Passolig complaints
    passolig_base_url = 'https://www.sikayetvar.com/passolig?page='
    scrape_all_pages(passolig_base_url, start_page=1, end_page=100)

    driver.quit()
