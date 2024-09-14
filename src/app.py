# app.py
# Flask uygulamasını başlatan ana dosya.
# API endpoint'lerini burada tanımlayabiliriz.

from flask import Flask, request, jsonify
from src.nlp.predict import PredictionService
from src.nlp.preprocess import preprocess_text

app = Flask(__name__)
service = PredictionService()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    contractions = {
        "değil": "değildir",
        "bişey": "bir şey",
        "diil": "değildir"
    }
    
    preprocessed_text = preprocess_text(text, contractions, turkish_stopwords=set(), to_english=True)
    
    predictions = service.predict_all(preprocessed_text)
    
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)