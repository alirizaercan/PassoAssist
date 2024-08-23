# predict.py
# Bu dosya eğitilmiş model ile tahmin yapma işlemlerini içerir.
# Yeni veriler üzerinde model tahminleri gerçekleştirir.

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

def predict(text, model):
    """
    Model ile tahmin yapar.
    :param text: Tahmin yapılacak metin
    :param model: Eğitilmiş model
    :return: Tahmin sonucu
    """
    return model.predict([text])[0]

if __name__ == "__main__":
    model = load_model('data/models/sentiment_model.pkl')
    text = "Bu bir örnek metin."
    prediction = predict(text, model)
    print(f"Tahmin Sonucu: {prediction}")
