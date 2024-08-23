# preprocessing.py
# Bu dosya veri ön işleme işlemlerini içerir.
# Örneğin: stop words çıkarma, lemmatization, veri temizleme işlemleri.

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """Bu fonksiyon metni temizler ve ön işler."""
    # Stop words çıkarma
    stop_words = set(stopwords.words('turkish'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
