import unittest
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from test_nlp import (
    correct_spelling, find_entities, predict_entity, predict_konu, predict_sentiment, 
    predict_severity, predict_multilabel, predict_all_models
)

# Paths to the models (adjust these paths as per your environment)
entity_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\entity_model.joblib'
konu_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\konu_model.joblib'
sentiment_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\sentiment\saribasmetehan_sentiment_model'
severity_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\severity_classifier.joblib'
multilabel_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\multilabel\multilabelclassifier.joblib'

# Test class for the NLP functions
class TestNLPModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load models once for all tests
        cls.entity_model = joblib.load(entity_model_path)
        cls.konu_model = joblib.load(konu_model_path)
        cls.severity_model, cls.tfidf_vectorizer = joblib.load(severity_model_path)
        cls.multilabel_model, cls.tfidf_vectorizer_multilabel = joblib.load(multilabel_model_path)
        cls.tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
        cls.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

    def test_correct_spelling(self):
        input_text = "passolg krt"
        expected_output = "passolig kart"
        self.assertEqual(correct_spelling(input_text), expected_output)

    def test_find_entities(self):
        input_text = "benim passolig kartim ile ilgili yardima ihtiyacim var"
        entities = ["passo", "passolig", "passolig kart"]
        expected_output = "passo; passolig; passolig kart"  # En spesifik entity döndürülüyor
        self.assertEqual(find_entities(input_text, entities), expected_output)
        
    def test_predict_entity(self):
        input_text = "passolig kart calismiyor"
        expected_output = "passo; passolig; passolig kart"
        self.assertEqual(predict_entity(input_text), expected_output)

    def test_predict_konu(self):
        input_text = "biletimi iptal etmek istiyorum"
        expected_label = 'iptal'  # Örneğin, iptal konusu
        self.assertEqual(predict_konu(input_text), expected_label)

    def test_predict_sentiment(self):
        input_text = "bu hizmetten cok memnun degilim"
        sentiment_label, sentiment_confidence = predict_sentiment(input_text)
        self.assertEqual(sentiment_label, 'olumsuz')  # Olumsuz duygu
        self.assertTrue(0 <= sentiment_confidence <= 1)  # Güvenin geçerliliğini kontrol et

    def test_predict_severity(self):
        input_text = "bu sorun hemen cozulmeli"
        severity_prediction, action_status, action_message = predict_severity(input_text)
        self.assertIn(severity_prediction, [0, 1, 2])  # Geçerli ciddiyet etiketlerini kontrol et
        self.assertIn(action_status, [0, 1])  # Eylem durumu 0 veya 1 olmalı

    def test_predict_multilabel(self):
        input_text = "odeme ve passolig kartim ile ilgili sorunlar yasiyorum"
        multilabel_prediction = predict_multilabel(input_text)
        self.assertIn('odeme', multilabel_prediction)  # Ödeme konusu tanınmalı
        self.assertIn('passolig kart', multilabel_prediction)

    def test_predict_all_models(self):
        input_text = "passolig kartimi iptal etmek istiyorum ve acil yardima ihtiyacim var"
        results = predict_all_models(input_text)
        
        # Her modelin sonucunu doğrula
        self.assertIn(results['Entity'], ['passo; passolig; passolig kart'])
        self.assertIn(results['Konu'], ['iptal', 'uyelik'])  # Örneğin: iptal veya üyelik
        self.assertIn(results['Sentiment']['label'], ['olumlu', 'notr', 'olumsuz'])
        self.assertIn(results['Severity']['severity_label'], [0, 1, 2])

        entity_prediction = predict_entity(input_text)
        print(f"Entity Prediction: {entity_prediction}")

if __name__ == "__main__":
    unittest.main()
