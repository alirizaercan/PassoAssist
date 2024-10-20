# preprocessing.py
# Bu dosya veri ön işleme işlemlerini içerir.
# Örneğin: stop words çıkarma, lemmatization, veri temizleme işlemleri.

import pandas as pd
import string
import re
from nltk.corpus import stopwords

# Stopword'leri yükleyelim
import nltk
nltk.download('stopwords')
turkish_stopwords = set(stopwords.words('turkish'))

# Türkçe karakterlerin normalizasyonu ve İngilizce karakterlere dönüşümü için fonksiyon
def normalize_turkish_chars(text, to_english=False):
    if to_english:
        turkish_char_map = str.maketrans(
            "çğıöşüÇĞİÖŞÜ",  # Türkçe karakterler
            "cgiosuCGIOSU"   # İngilizce karşılıkları
        )
    else:
        turkish_char_map = str.maketrans(
            "ÇĞİÖŞÜçğıöşü",  # Türkçe karakterler
            "CGIOSUcgiosu"   # Karşılıkları küçük olacak şekilde ayarlandı
        )
    
    text = text.translate(turkish_char_map)
    
    # Küçük harfe çevirme işlemi tüm harfler için
    return text.lower()

# Belirli semboller ve boşlukları kaldırma fonksiyonu
def remove_unwanted_chars(text):
    unwanted_patterns = [
        r"http\S+",  # HTTP ve URL'leri kaldırma
        r"@\S+",  # Mention'ları kaldırma
        r"#\S+",  # Hashtag'leri kaldırma
        r"[0-9]",  # Sayıları kaldırma
        r"\W",  # Özel karakterleri kaldırma
        r"\b\w\b",  # Tek karakterlik kelimeleri kaldırma
        r"\s+"  # Fazla boşlukları kaldırma
    ]
    
    # Her istenmeyen ifadeyi sırayla kaldır
    for pattern in unwanted_patterns:
        text = re.sub(pattern, ' ', text)

    return text.strip()

# Metin işleme fonksiyonu
def preprocess_text(text, contractions, stopwords, to_english=False):
    if not isinstance(text, str):
        return text  # Eğer metin bir string değilse olduğu gibi döndür
    
    # Türkçe karakterleri normalize etme ve küçük harfe çevirme
    text = normalize_turkish_chars(text, to_english)
    text = text.replace('İ', 'i').lower()
    
    # Gereksiz semboller, URL'ler, hashtag'ler, mention'lar, sayılar vs. temizleniyor
    text = remove_unwanted_chars(text)
    
    # Fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stopword'leri kaldırma
    text = ' '.join([word for word in text.split() if word not in stopwords])
    
    # Contraction'ları genişletme (eğer contraction sözlüğü mevcutsa)
    text = ' '.join([contractions.get(word, word) for word in text.split()])

    return text

# Veri yükleme ve işleme
def preprocess_data(input_file, output_file, contractions, to_english=False):
    # Veriyi yükle
    df = pd.read_csv(input_file)
    
    # 'text' sütunundaki metinleri işlemden geçir
    df['text'] = df['text'].apply(lambda x: preprocess_text(x, contractions, turkish_stopwords, to_english))
    
    # İşlenmiş veriyi kaydet
    df.to_csv(output_file, index=False)
    print(f"İşlenmiş veri '{{output_file}}' dosyasına kaydedildi.")

if __name__ == "__main__":
    input_file = 'data/raw/raw_data_passo.csv'
    output_file = 'data/test/test_processed_data.csv'
    
    # Contraction sözlüğü (genişletilebilir)
    contractions = {
        "değil": "değildir",
        "bişey": "bir şey",
        "diil": "değildir"
        # Buraya eklemeler yapabilirsiniz.
    }
    
    # İngilizce karakterlere dönüşüm yapmak istiyorsanız to_english=True olarak ayarlayın
    preprocess_data(input_file, output_file, contractions, to_english=True)
