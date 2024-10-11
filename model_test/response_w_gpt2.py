import pandas as pd
import joblib
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer

# Model ve tokenizer'ları yükle
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise Exception(f"Model '{path}' yüklenirken hata oluştu: {e}")

gpt2_model_name = "cenkersisman/gpt2-turkish-128-token"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

turna_tokenizer = AutoTokenizer.from_pretrained("boun-tabi-LMG/TURNA")
turna_model = AutoModelForSeq2SeqLM.from_pretrained("boun-tabi-LMG/TURNA")

# 'Konu' için label değerleri
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

# Modelleri yükle
entity_model = load_model(entity_model_path)
konu_model = load_model(konu_model_path)

# Severity model ve Multilabel model tuple olabilir
severity_model = load_model(severity_model_path)
if isinstance(severity_model, tuple):
    severity_model = severity_model[0]  # Modeli tuple'dan çıkarıyoruz

multilabel_model, tfidf_vectorizer = load_model(multilabel_model_path)
if isinstance(multilabel_model, tuple):
    multilabel_model = multilabel_model[0]  # Multilabel modeli tuple'dan çıkarıyoruz

sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

# Fonksiyonlar
def correct_spelling(sentence):
    corrections = {
        "passolg": "passolig",
        "passolig krt": "passolig kart",
    }
    words = sentence.split()
    return ' '.join([corrections.get(word.lower(), word) for word in words])

def find_entities(sentence, entity_list):
    corrected_sentence = correct_spelling(sentence)
    found_entities = [entity for entity in entity_list if re.search(rf'\b{entity}\b', corrected_sentence, re.IGNORECASE)]
    return found_entities if found_entities else ["No entity found."]

def predict_entity(input_text):
    entities = ["passo", "passolig", "passolig kart"]
    return "; ".join(find_entities(input_text, entities))

def predict_konu(input_text):
    predicted_label = konu_model.predict([input_text])[0]
    return konu_labels[predicted_label]

def predict_sentiment(input_text):
    inputs = sentiment_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    sentiment_logits = sentiment_model(**inputs).logits
    sentiment_prediction = sentiment_logits.argmax().item()
    predicted_probs = torch.softmax(sentiment_logits, dim=1)[0]
    
    label_mapping = {0: 'notr', 1: 'olumlu', 2: 'olumsuz'}
    return label_mapping[sentiment_prediction], predicted_probs[sentiment_prediction].item()

def predict_severity(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    severity_prediction = severity_model.predict(input_tfidf)[0]
    
    if severity_prediction == 2:
        return severity_prediction, 1, "Acil Harekete Geçin!"
    elif severity_prediction == 1:
        return severity_prediction, 1, "Aksiyon Almanız Önerilir."
    return severity_prediction, 0, "Harekete Geçmeye Gerek Yok."

def predict_multilabel(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    multilabel_prediction = multilabel_model.predict(input_tfidf)[0]
    
    labels = ['bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig kart', 'diger']
    return dict(zip(labels, multilabel_prediction))

def predict_all_models(input_text):
    return {
        'Entity': predict_entity(input_text),
        'Konu': predict_konu(input_text),
        'Sentiment': {
            'label': predict_sentiment(input_text)[0],
            'confidence': predict_sentiment(input_text)[1]
        },
        'Severity': {
            'severity_label': predict_severity(input_text)[0],
            'action_status': predict_severity(input_text)[1],
            'action_message': predict_severity(input_text)[2]
        },
        'Multilabel': predict_multilabel(input_text)
    }

# Kurumsal dilde yanıt oluşturma
def generate_professional_response(data):
    response = []

    # Ciddiyet ve aksiyon
    if data['Severity']['severity_label'] == 2:
        response.append("Durumunuz aciliyet arz etmektedir. Ekiplerimiz en kısa sürede gerekli adımları atacaktır.")
    elif data['Severity']['severity_label'] == 1:
        response.append("Sorununuz öncelikli olarak değerlendirilmektedir. Gerekli aksiyonlar kısa süre içinde alınacaktır.")
    else:
        response.append("Sorununuz şu an için acil bir müdahale gerektirmemektedir. Ancak önerilerimiz doğrultusunda işlemlerinizi gözden geçirmenizi tavsiye ederiz.")

    # Sentiment durumu
    if data['Sentiment']['label'] == 'olumsuz':
        response.append("Yaşamış olduğunuz aksaklık için özür dileriz. Sorununuzu çözmek adına aşağıdaki adımları izleyebilirsiniz:")
        if data['Multilabel']['odeme'] == 1:
            response.append("Ödeme işleminiz sırasında bir sorun yaşandığını anlıyoruz. Lütfen ödeme yöntemlerinizi kontrol edin ve işlem bilgilerini doğru girdiğinizden emin olun. Eğer sorun devam ederse, müşteri hizmetlerimizle iletişime geçmekten çekinmeyin.")
        # Uygulama ile ilgili sorun
        if data['Multilabel']['uygulama'] == 1:
            response.append("Kullandığınız uygulamanın en güncel sürümünü kullanıp kullanmadığınızı kontrol edebilirsiniz. Ayrıca, uygulamayı yeniden başlatmak sorunu çözebilir. Eğer sorun devam ederse, teknik destek ekibimiz size yardımcı olmaktan memnuniyet duyacaktır.")
        # Bilet ile ilgili sorun
        if data['Multilabel']['bilet'] == 1:
            response.append("Bilet işlemlerinizde bir aksaklık olduğunu fark ettik. Lütfen bilet bilgilerinizin doğru olduğunu ve geçerli bir işlem gerçekleştirdiğinizi kontrol edin. Yardıma ihtiyaç duyarsanız, müşteri hizmetleri ekibimiz size destek sağlayacaktır.")
        # Müşteri hizmetleri ile ilgili sorun
        if data['Multilabel']['musteri_hizmetleri'] == 1:
            response.append("Müşteri hizmetlerimizle ilgili yaşadığınız deneyimle ilgili üzgünüz. Size en iyi hizmeti sunmak için çalışıyoruz. Sorununuzun çözümü için gerekli adımları atacağımızdan emin olabilirsiniz.")
        # Passolig ile ilgili sorun
        if data['Multilabel']['passolig'] == 1:
            response.append("Passolig işleminiz sırasında bir sorun yaşadığınızı görüyoruz. Lütfen Passolig kartınızın aktif ve doğru bilgilerle işlendiğini kontrol edin. Sorunun devam etmesi durumunda destek ekibimizle iletişime geçebilirsiniz.")
        # Passolig kart ile ilgili sorun
        if data['Multilabel']['passolig kart'] == 1:
            response.append("Passolig kartınızla ilgili yaşadığınız sıkıntıyı anlıyoruz. Lütfen kartınızın geçerlilik süresini ve ödeme işlemlerini kontrol edin. Herhangi bir sorun devam ederse, müşteri hizmetlerimizle iletişime geçebilirsiniz.")
        # Diğer genel sorunlar
        if data['Multilabel']['diger'] == 1:
            response.append("Yaşadığınız diğer sorunlar hakkında size en kısa sürede yardımcı olacağız. Daha fazla bilgi veya destek almak için lütfen müşteri hizmetlerimizle iletişime geçin.")
        
    elif data['Sentiment']['label'] == 'notr':
        response.append("Geri bildiriminiz doğrultusunda, süreçlerinizi dikkatle takip ettiğinizi öneriyoruz.")
    else:
        response.append("Olumlu geri bildiriminiz için teşekkür ederiz. Sizlere daha iyi hizmet verebilmek adına çalışmalarımıza devam edeceğiz.")

    # Konu ile ilgili geri bildirim
    response.append(f"Belirtilen şikayet konusu: {data['Konu']}. Gerekli departmanlarımız tarafından en kısa sürede çözüme kavuşturulacaktır.")
    
    # Entity durumu
    if data['Entity'] != "No entity found.":
        response.append(f"{data['Entity']} ile ilgili işlemler başlatılmıştır. Tarafınıza bilgilendirme yapılacaktır.")

    return " ".join(response)

# Ana program döngüsü
def generate_gpt2_response(user_input, data):
    prompt = f"Şikayet: {user_input}\n"
    
    # Analiz sonuçlarını prompt'a ekle
    prompt += f"Konu: {data['Konu']}\n"
    prompt += f"Sentiment: {data['Sentiment']['label']}\n"
    prompt += f"Severity: {data['Severity']['severity_label']} (Aksiyon: {data['Severity']['action_message']})\n"
    prompt += "Lütfen bu duruma uygun kurumsal bir yanıt oluştur: "
    
    # GPT-2 modelinden yanıt üret
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response_text

# Ana program döngüsü
def generate_turna_response(user_input, data):
    prompt = f"Şikayet: {user_input}\n"
    
    # Analiz sonuçlarını prompt'a ekle
    prompt += f"Konu: {data['Konu']}\n"
    prompt += f"Sentiment: {data['Sentiment']['label']}\n"
    prompt += f"Severity: {data['Severity']['severity_label']} (Aksiyon: {data['Severity']['action_message']})\n"
    prompt += "Lütfen bu duruma uygun kurumsal bir yanıt oluştur: "
    
    # TURNA modelinden yanıt üret
    inputs = turna_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = turna_model.generate(**inputs, max_length=150, num_return_sequences=1)
    turna_response = turna_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return turna_response

# Ana program döngüsü
if __name__ == "__main__":
    user_input = input("Lütfen şikayetinizi girin: ")
    
    # Tüm analiz sonuçlarını al
    data = predict_all_models(user_input)
    
    # GPT-2 modeliyle yanıt oluştur
    gpt2_response = generate_gpt2_response(user_input, data)
    
    # TURNA modeliyle yanıt oluştur
    turna_response = generate_turna_response(user_input, data)
    
    # Yanıtı ve GPT-2 + TURNA çıktısını birleştir
    response = generate_professional_response(data)
    response += "\n\nGPT-2 yanıtı:\n" + gpt2_response
    response += "\n\nTURNA yanıtı:\n" + turna_response
    
