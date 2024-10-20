# routes.py
# Bu dosya API endpoint'lerini içerir.
# Web scraping, sentiment analizi, veri kategorilendirme gibi işlemler için yollar tanımlar.

import os
import json
import pandas as pd
from flask import Blueprint, request, jsonify
from scripts.scraping import scrape_all_pages 
from scripts.model_training_scripts.text_cleaning import preprocess_text, load_contractions, load_stopwords
from scripts.model_training_scripts.test_sentiment_model import load_model, predict_sentiment

api = Blueprint('api', __name__)

# Dosya yolları
SCRAPED_DATA_PATH = 'data/json/scraped_data.json'
CLEANED_DATA_PATH = 'data/json/cleaned_data.json'
ANALYZED_DATA_PATH = 'data/json/analyzed_data.json'

# Load model and tokenizer once for reuse (from the imported module)
tokenizer, sentiment_model = load_model('data/models/sentiment/saribasmetehan_sentiment_model')

@api.route('/scrape', methods=['GET'])
def scrape():
    base_url = request.args.get('url')
    start_page = int(request.args.get('start_page', 1))
    end_page = int(request.args.get('end_page', 1))  # Default to scrape only one page if not provided
    keyword_filter = request.args.get('keyword_filter', None)  # Optional filter

    # Scrape the pages
    complaints = scrape_all_pages(base_url, start_page=start_page, end_page=end_page, keyword_filter=keyword_filter)

    # Verileri JSON formatında kaydet
    with open(SCRAPED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump({"complaints": complaints}, f, ensure_ascii=False, indent=4)

    # Scrape edilen verileri JSON olarak dön
    return jsonify({"scraped_data": complaints})

@api.route('/clean_data', methods=['POST'])
def clean_data():
    # Load the necessary files for contractions and stopwords
    contractions_file = 'data/json/contractions.json'
    stopwords_file = 'data/json/stopwords.json'
    
    # Load contractions and stopwords
    contractions = load_contractions(contractions_file)
    stopwords = load_stopwords(stopwords_file)

    # Scrape edilen veriyi yükle
    with open(SCRAPED_DATA_PATH, 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)['complaints']

    # DataFrame'e dönüştür
    df = pd.DataFrame(scraped_data)
    
    # Veriyi temizleme işlemi
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, contractions, stopwords))

    # Temizlenmiş verileri JSON olarak kaydet
    cleaned_data = df.to_dict(orient='records')
    with open(CLEANED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump({"cleaned_data": cleaned_data}, f, ensure_ascii=False, indent=4)

    # Temizlenmiş verileri JSON olarak dön
    return jsonify({"cleaned_data": cleaned_data})

@api.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    # Temizlenmiş veriyi yükle
    with open(CLEANED_DATA_PATH, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)['cleaned_data']

    # Sentiment analizi yap ve sonuçları sakla
    analyzed_data = []
    for entry in cleaned_data:
        text = entry['clean_text']
        sentiment, confidence = predict_sentiment(sentiment_model, tokenizer, text)
        analyzed_data.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 2)
        })

    # Analiz edilen verileri JSON olarak kaydet
    with open(ANALYZED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump({"analyzed_data": analyzed_data}, f, ensure_ascii=False, indent=4)

    # Analiz edilen verileri JSON olarak dön
    return jsonify({"analyzed_data": analyzed_data})
