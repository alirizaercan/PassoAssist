# predict.py
# Bu dosya eğitilmiş model ile tahmin yapma işlemlerini içerir.
# Yeni veriler üzerinde model tahminleri gerçekleştirir.

# predict.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class PredictionService:
    def __init__(self):
        self.models = {
            'severity': BertForSequenceClassification.from_pretrained('data/models/severity_model'),
            'bilet': BertForSequenceClassification.from_pretrained('data/models/bilet_model'),
            'musteri_hizmetleri': BertForSequenceClassification.from_pretrained('data/models/musteri_hizmetleri_model'),
            'odeme': BertForSequenceClassification.from_pretrained('data/models/odeme_model'),
            'uygulama': BertForSequenceClassification.from_pretrained('data/models/uygulama_model'),
            'passolig': BertForSequenceClassification.from_pretrained('data/models/passolig_model'),
            'passolig_kart': BertForSequenceClassification.from_pretrained('data/models/passolig_kart_model'),
            'diger': BertForSequenceClassification.from_pretrained('data/models/diger_model')
        }
        self.tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

    def predict(self, text, column_name):
        model = self.models[column_name]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.item()

    def predict_all(self, text):
        predictions = {}
        columns = ['severity', 'bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig_kart', 'diger']
        severity_prediction = self.predict(text, 'severity')
        predictions['severity'] = severity_prediction
        predictions['aksiyon'] = 1 if severity_prediction in [1, 2] else 0
        for column in columns[1:]:
            predictions[column] = self.predict(text, column)
        return predictions

if __name__ == "__main__":
    service = PredictionService()
    text = "Örnek bir metin."
    print(service.predict_all(text))
