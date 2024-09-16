from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
from text_cleaning import preprocess_text, load_contractions, load_stopwords

app = FastAPI()

# Model ve tokenizer yolları
model_path = 'data/models/bert_sentiment_model'  # BERT modelinizin yolu

# Model ve tokenizer yükleme
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Contractions ve stopwords yolları
contractions_path = 'C:/Users/Ali Riza Ercan/Documents/GitHub/PassoAssist/model_test/contractions.json'
stopwords_path = 'C:/Users/Ali Riza Ercan/Documents/GitHub/PassoAssist/model_test/stopwords.json'

# Contractions ve stopwords yükleme
contractions = load_contractions(contractions_path)
stopwords = load_stopwords(stopwords_path)

# Kullanıcıdan gelecek veri yapısı
class Text(BaseModel):
    text: str

# POST isteği ile duygu analizi tahmini yapan endpoint
@app.post("/predict/")
def predict_sentiment(text: Text):
    try:
        # Metni temizleme
        cleaned_text = preprocess_text(text.text, contractions, stopwords)
        print(f"Cleaned text: {cleaned_text}")  # Debugging amaçlı

        # Metni token haline getirme ve modele uygun hale getirme
        inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Modeli kullanarak tahmin yapma
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            sentiment = torch.argmax(probabilities).item()
            score = probabilities[0][sentiment].item()

        # Üç sınıflı sentiment (olumlu, nötr, olumsuz)
        sentiment_labels = ['olumsuz', 'notr', 'olumlu']
        
        return {
            "cleaned_text": cleaned_text,
            "sentiment": sentiment_labels[sentiment],
            "score": score
        }
    except Exception as e:
        print(f"Error: {str(e)}")  # Hata ayıklama
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="debug")
