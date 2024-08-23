# train_model.py
# Bu dosya NLP modelinin eğitimini içerir.
# Eğitilmiş model verilerle burada oluşturulur.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

def train_model(df):
    """
    Modeli eğitir.
    :param df: Eğitim verisi içeren DataFrame
    :return: Eğitilmiş model
    """
    X = df['text']
    y = df['label']
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    model.fit(X, y)
    
    return model

if __name__ == "__main__":
    df = pd.read_csv('data/processed/example_processed_data.csv')
    model = train_model(df)
    with open('data/models/sentiment_model.pkl', 'wb') as file:
        pickle.dump(model, file)
