# sentiment_analysis.py
# Bu dosya sentiment analiz işlemlerini içerir.
# Eğitilmiş modeli kullanarak metinlerin duygu durumlarını belirler.

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def load_model(model_path):
    """
    Eğitilmiş modeli yükler.
    :param model_path: Model dosya yolu
    :return: Yüklenen model
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def analyze_sentiment(text, model):
    """
    Metnin sentiment analizini yapar.
    :param text: Analiz edilecek metin
    :param model: Eğitilmiş model
    :return: Metnin duygu durumu
    """
    return model.predict([text])[0]

if __name__ == "__main__":
    model = load_model('data/models/sentiment_model.pkl')
    text = "Bu bir örnek metin."
    sentiment = analyze_sentiment(text, model)
    print(f"Metnin Sentiment Durumu: {sentiment}")
