# routes.py
# Bu dosya API endpoint'lerini içerir.
# Web scraping, sentiment analizi, veri kategorilendirme gibi işlemler için yollar tanımlar.

from flask import Blueprint, request, jsonify
from scripts.scraping_passo import scrape_data
from scripts.sentiment_analysis import analyze_sentiment, load_model
from scripts.categorize import categorize_complaint

api = Blueprint('api', __name__)
model = load_model('data/models/sentiment_model.pkl')

@api.route('/scrape', methods=['GET'])
def scrape():
    url = request.args.get('url')
    data = scrape_data(url)
    return jsonify({"data": str(data)})

@api.route('/analyze_sentiment', methods=['POST'])
def analyze():
    text = request.json.get('text')
    sentiment = analyze_sentiment(text, model)
    return jsonify({"sentiment": sentiment})

@api.route('/categorize', methods=['POST'])
def categorize():
    text = request.json.get('text')
    category = categorize_complaint(text)
    return jsonify({"category": category})
