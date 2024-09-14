# preprocess.py
# Bu dosya metinlerin ön işlenmesini içerir.
# NLP işlemleri için verileri temizler ve hazırlar.

# src/nlp/preprocess.py
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
turkish_stopwords = set(stopwords.words('turkish'))

def normalize_turkish_chars(text, to_english=False):
    if to_english:
        turkish_char_map = str.maketrans(
            "çğıöşüÇĞİÖŞÜ",  # Türkçe karakterler
            "cgiosuCGIOSU"   # İngilizce karşılıkları
        )
    else:
        turkish_char_map = str.maketrans(
            "çğıöşü",  # Türkçe karakterler
            "cgiosu"   # İngilizce karşılıkları
        )
    return text.translate(turkish_char_map)

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

def preprocess_text(text, contractions, stopwords, to_english=False):
    if not isinstance(text, str):
        return text  # Eğer metin bir string değilse olduğu gibi döndür
    
    text = normalize_turkish_chars(text, to_english)
    text = text.replace('İ', 'i').lower()
    text = remove_unwanted_chars(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ' '.join([contractions.get(word, word) for word in text.split()])
    return text
