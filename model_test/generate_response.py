import pandas as pd
import joblib  # joblib kütüphanesini kullanıyoruz
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# 'konu' değerleri
konu_labels = [
    'cagri merkezi yetkinlik', 'diger', 'genel', 'odeme', 'uygulama',
    'iptal', 'degisiklik', 'uyelik', 'iade', 'transfer', 'fatura'
]

# Model yolları
entity_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\entity_model.joblib'
konu_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\konu_model.joblib'
sentiment_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\sentiment\saribasmetehan_sentiment_model'
severity_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\severity_classifier.joblib'
multilabel_model_path = r'C:\Users\Ali Riza Ercan\Desktop\Data Science\PassoAssist\PassoAssist\data\models\multilabel\multilabelclassifier.joblib'
llm_model_path = 'redrussianarmy/gpt2-turkish-cased'  # Güncellenmiş model

# Modelleri yükle
def load_model(path):
    return joblib.load(path)

try:
    entity_model = load_model(entity_model_path)
except Exception as e:
    print(f"Entity model yüklenirken hata oluştu: {e}")

try:
    konu_model = load_model(konu_model_path)
except Exception as e:
    print(f"Konu model yüklenirken hata oluştu: {e}")

try:
    loaded_model = load_model(severity_model_path)
    if isinstance(loaded_model, tuple):
        severity_model = loaded_model[0]
    else:
        severity_model = loaded_model
except Exception as e:
    print(f"Severity model yüklenirken hata oluştu: {e}")

try:
    multilabel_model, tfidf_vectorizer = load_model(multilabel_model_path)
except Exception as e:
    print(f"Multilabel model yüklenirken hata oluştu: {e}")

try:
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
except Exception as e:
    print(f"Sentiment model yüklenirken hata oluştu: {e}")

# LLM Modelini Yükle
try:
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path)
except Exception as e:
    print(f"LLM model yüklenirken hata oluştu: {e}")

# Entity tahmini için gerekli fonksiyonlar
def correct_spelling(sentence):
    corrected_spellings = {
        "passolg": "passolig",
        "passolig krt": "passolig kart",
    }
    words = sentence.split()
    corrected_words = [corrected_spellings.get(word.lower(), word) for word in words]
    return ' '.join(corrected_words)

def find_entities(sentence, entity_list):
    corrected_sentence = correct_spelling(sentence)
    found_entities = []
    for entity in entity_list:
        pattern = rf'\b{entity}(?:\w+)?\b'
        if re.search(pattern, corrected_sentence, re.IGNORECASE):
            found_entities.append(entity)
    return found_entities

def process_sentence(sentence, entity_list):
    found_entities = find_entities(sentence, entity_list)
    if found_entities:
        return "; ".join(found_entities)
    return "No entity found."

def predict_entity(input_text):
    entities = ["passo", "passolig", "passolig kart"]
    return process_sentence(input_text, entities)

def predict_konu(input_text):
    predicted_label = konu_model.predict([input_text])
    return konu_labels[predicted_label[0]]

def predict_sentiment(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        sentiment_logits = sentiment_model(**inputs).logits
        sentiment_prediction = sentiment_logits.argmax().item()
        predicted_probs = torch.softmax(sentiment_logits, dim=1)[0]

    label_mapping = {1: 'olumlu', 0: 'notr', 2: 'olumsuz'}
    sentiment_label = label_mapping[sentiment_prediction]
    sentiment_confidence = predicted_probs[sentiment_prediction].item()

    return sentiment_label, sentiment_confidence

def predict_severity(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    severity_prediction = severity_model.predict(input_tfidf)[0]

    if severity_prediction == 2:
        action_status = 1
        action_message = "Acil Harekete Geçin!"
    elif severity_prediction == 1:
        action_status = 1
        action_message = "Aksiyon Almanız Önerilir."
    else:
        action_status = 0
        action_message = "Harekete Geçmeye Gerek Yok."
    
    return severity_prediction, action_status, action_message

def predict_multilabel(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    multilabel_prediction = multilabel_model.predict(input_tfidf)

    return {
        'bilet': multilabel_prediction[0][0],
        'musteri_hizmetleri': multilabel_prediction[0][1],
        'odeme': multilabel_prediction[0][2],
        'uygulama': multilabel_prediction[0][3],
        'passolig': multilabel_prediction[0][4],
        'passolig kart': multilabel_prediction[0][5],
        'diger': multilabel_prediction[0][6],
    }

# Yeni bir yanıt oluşturma fonksiyonu
def generate_response(input_text):
    inputs = llm_tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = llm_model.generate(inputs, max_length=100)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def predict_all_models(input_text):
    entity_prediction = predict_entity(input_text)
    konu_prediction = predict_konu(input_text)
    sentiment_prediction, sentiment_confidence = predict_sentiment(input_text)
    severity_prediction, action_status, action_message = predict_severity(input_text)
    multilabel_prediction = predict_multilabel(input_text)
    llm_response = generate_response(input_text)

    return {
        'Entity': entity_prediction,
        'Konu': konu_prediction,
        'Sentiment': {
            'label': sentiment_prediction,
            'confidence': sentiment_confidence
        },
        'Severity': {
            'severity_label': severity_prediction,
            'action_status': action_status,
            'action_message': action_message
        },
        'Multilabel': multilabel_prediction,
        'LLM_Response': llm_response  # Yeni yanıt
    }

# Ana çalışma döngüsü
if __name__ == "__main__":
    # Var olan verileri yükle ya da yeni bir DataFrame oluştur
    try:
        df = pd.read_csv('user_input.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['text', 'entity', 'sentiment', 'konu', 'severity', 
                                   'bilet', 'musteri_hizmetleri', 'odeme', 
                                   'uygulama', 'passolig', 'passolig kart', 
                                   'diger', 'aksiyon', 'llm_response'])

    while True:
        test_input = input("Bir metin girin (çıkmak için 'q' yazın): ")
        if test_input.lower() == 'q':
            break
        
        # Modellerden tahmin al
        results = predict_all_models(test_input)

        # Sonuçları terminalde göster
        print("\nTahmin Sonuçları:")
        for model, prediction in results.items():
            if model == 'Sentiment':
                print(f"{model}: {prediction['label']} (Güven: {prediction['confidence']:.2f})")
            elif model == 'Severity':
                print(f"{model}: Severity {prediction['severity_label']}, Aksiyon Durumu: {prediction['action_status']} ({prediction['action_message']})")
            elif model == 'LLM_Response':
                print(f"{model}: {prediction}")
            else:
                print(f"{model}: {prediction}")

        # Tahmin sonuçlarını DataFrame'e ekle
        new_row = pd.DataFrame([{
            'text': test_input,
            'entity': results['Entity'],
            'sentiment': results['Sentiment']['label'],
            'konu': results['Konu'],
            'severity': results['Severity']['severity_label'],
            'bilet': results['Multilabel']['bilet'],
            'musteri_hizmetleri': results['Multilabel']['musteri_hizmetleri'],
            'odeme': results['Multilabel']['odeme'],
            'uygulama': results['Multilabel']['uygulama'],
            'passolig': results['Multilabel']['passolig'],
            'passolig kart': results['Multilabel']['passolig kart'],
            'diger': results['Multilabel']['diger'],
            'aksiyon': results['Severity']['action_status'],
            'llm_response': results['LLM_Response']
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    
    # DataFrame'i CSV dosyası olarak kaydet
    df.to_csv('user_input.csv', index=False)
