import time
import csv
from selenium import webdriver
from bs4 import BeautifulSoup

# Selenium setup
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Run in headless mode (no browser window)
driver = webdriver.Chrome(options=options)

# Target URL
base_url = 'https://www.sikayetvar.com/passo?page='

# CSV file setup
csv_file = 'data/raw/passo_topic_title.csv'
fields = ['text', 'bilet_iade', 'giris_sorun', 'musteri_hiz', 'odeme_sorun', 'uygulama_hata', 'diger', 'severity', 'konu', 'aksiyon', 'sentiment']

def scrape_page(page_url):
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
            complaints.append({
                'text': description,
                'bilet_iade': '',  # Placeholder for future data
                'giris_sorun': '',  # Placeholder for future data
                'musteri_hiz': '',  # Placeholder for future data
                'odeme_sorun': '',  # Placeholder for future data
                'uygulama_hata': '',  # Placeholder for future data
                'diger': '',  # Placeholder for future data
                'severity': '',  # Placeholder for future data
                'konu': '',  # Placeholder for future data
                'aksiyon': '',  # Placeholder for future data
                'sentiment': ''  # Placeholder for future sentiment analysis
            })
    
    return complaints

def scrape_all_pages():
    start_page = 1
    end_page = 71  # Specify the end page number
    all_complaints = []

    for page_num in range(start_page, end_page + 1):
        page_url = base_url + str(page_num)
        print(f"Scraping page {page_num}: {page_url}")

        complaints = scrape_page(page_url)
        
        if not complaints:
            break  # Stop if no complaints found (reached the end)
        
        all_complaints.extend(complaints)

    # Save to CSV
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_complaints)

    print(f"Scraping complete. Total complaints: {len(all_complaints)}")

if __name__ == "__main__":
    scrape_all_pages()
    driver.quit()
