from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os

# Selenium için Chrome ayarlarını yap
options = Options()
# options.add_argument('--headless')  # Tarayıcıyı başsız modda çalıştır

# WebDriver yolu (ChromeDriver)
service = Service("..\\chromedriver\\chromedriver.exe")  # ChromeDriver yolunu belirtin
driver = webdriver.Chrome(service=service, options=options)

# Hedef URL
url = 'https://www.sikayetvar.com/passolig'

# Sayfayı aç
driver.get(url)

# Kategorileri içeren div'i seç
swiper_div = driver.find_element(By.CLASS_NAME, 'swiper-wrapper')

# Kategorileri bulmak için gerekli tagleri seç
categories = swiper_div.find_elements(By.CLASS_NAME, 'company-collections__container')

# Kategorileri liste olarak çek
category_list = [category.get_attribute('title') for category in categories]

# Kategorileri dosyaya kaydet
output_file_path = 'data/raw/passolig_topic_titles.txt'

# Klasör var mı kontrol et, yoksa oluştur
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Kategorileri dosyaya yaz
with open(output_file_path, 'w', encoding='utf-8') as file:
    for category in category_list:
        file.write(category + '\n')

print(f"Kategoriler '{output_file_path}' dosyasına kaydedildi.")

# WebDriver'ı kapat
driver.quit()
