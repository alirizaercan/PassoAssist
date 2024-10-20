import string
import re
import json
from nltk.corpus import stopwords
import nltk

# Stopword'leri yükleyelim
nltk.download('stopwords')

# Stopwords ve contractions'ı JSON dosyalarından yükleme
def load_stopwords(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return set(json.load(f))

def load_contractions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Türkçe karakterlerin normalizasyonu ve İngilizce karakterlere dönüşümü için fonksiyon
def normalize_turkish_chars(text, to_english=False):
    turkish_char_map = str.maketrans(
        "çğıöşüÇĞİÖŞÜ",  # Türkçe karakterler
        "cgiosuCGIOSU" if to_english else "cgiosu"
    )
    return text.translate(turkish_char_map)

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
    
    # Gereksiz semboller temizleniyor
    text = remove_unwanted_chars(text)
    
    # Fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stopword'leri kaldırma
    text = ' '.join([word for word in text.split() if word not in stopwords])
    
    # Contraction'ları genişletme
    text = ' '.join([contractions.get(word, word) for word in text.split()])

    return text

# Kullanıcıdan gelen veriyi işleyen fonksiyon
def process_user_input(user_input, contractions_json, stopwords_json, to_english=False):
    # JSON'dan contraction ve stopword'leri yükleme
    contractions = load_contractions(contractions_json)
    stopwords = load_stopwords(stopwords_json)
    
    # Kullanıcı girdisini temizle ve işleme
    cleaned_text = preprocess_text(user_input, contractions, stopwords, to_english)

    return cleaned_text
