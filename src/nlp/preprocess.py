# preprocess.py
# Bu dosya metinlerin ön işlenmesini içerir.
# NLP işlemleri için verileri temizler ve hazırlar.

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """
    Metni temizler ve ön işler.
    :param text: İşlenecek metin
    :return: Ön işlenmiş metin
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    cleaned_text = ' '.join([word for word in words if word.lower() not in stop_words])
    return cleaned_text

if __name__ == "__main__":
    df = pd.read_csv('data/raw/example_raw_data.csv')
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv('data/processed/example_processed_data.csv', index=False)
