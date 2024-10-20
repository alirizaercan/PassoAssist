import time
import csv
from selenium import webdriver
from bs4 import BeautifulSoup

# Selenium setup
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Run in headless mode (no browser window)
driver = webdriver.Chrome(options=options)

# Kısaltılmış CSV dosya yolu
csv_file = 'data/test/test_scrapping_data.csv'
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
    time.sleep(2)  # Sayfanın yüklenmesi için bekleme süresi

    # Sayfa kaynağını al ve BeautifulSoup ile parse et
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Şikayet yazılarını bul
    articles = soup.find_all('article', class_='card-v2 ga-v ga-c')

    complaints = []

    for article in articles:
        description_tag = article.find('p', class_='complaint-description')
        if description_tag:
            description = description_tag.get_text(strip=True).replace('...', '')
            if keyword_filter is None or keyword_filter in description:
                complaints.append({'text': description})

    return complaints

def scrape_test():
    page_url = 'https://www.sikayetvar.com/passo?page=1'  # Sadece ilk sayfayı çekiyoruz
    complaints = scrape_page(page_url, keyword_filter='Passo')

    # CSV'ye yaz
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(complaints)

    return complaints

if __name__ == "__main__":
    scrape_test()
    driver.quit()
