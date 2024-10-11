import pandas as pd

# Custom LLM yanıtlarını oluşturan fonksiyon
def generate_custom_response(entity, konu, sentiment, severity, multilabel):
    # 1. Senaryo: Ödeme sorunu + Olumsuz duygu
    if konu == 'odeme' and sentiment == 'olumsuz':
        llm_input = f"Kullanıcı ödeme konusunda sorun yaşıyor ve olumsuz bir deneyim yaşamış. Acil çözüm önerisi sunarak müşteri memnuniyetini nasıl sağlayabilirim?"

    # 2. Senaryo: Ödeme sorunu + Olumlu duygu (Sorun çözülmüş olabilir)
    elif konu == 'odeme' and sentiment == 'olumlu':
        llm_input = f"Kullanıcı ödeme sorunuyla karşılaştı ancak olumlu bir geri bildirimde bulunuyor. Sorun çözüldü mü yoksa ek bir yardım sunmam gerekiyor mu?"

    # Diğer senaryolar eklenebilir...
    else:
        llm_input = f"Konu: {konu}, Duygu: {sentiment}. Bu duruma nasıl yanıt vermeliyim?"

    # Yanıtı oluşturma
    return llm_input

# Ana fonksiyon
def process_data(df, test_input, model_predictions):
    # Model tahminlerini al ve LLM entegrasyonunu kullanarak yanıt oluştur
    entity = model_predictions['Entity']
    konu = model_predictions['Konu']
    sentiment = model_predictions['Sentiment']['label']
    severity = model_predictions['Severity']['severity_label']
    multilabel = model_predictions['Multilabel']
    
    # LLM yanıtını oluştur
    llm_input = generate_custom_response(entity, konu, sentiment, severity, multilabel)
    
    # Sonuçları yazdır
    print(f"Tahmin Sonuçları: {model_predictions}")
    print(f"LLM Yanıtı: {llm_input}")

    # Yeni veriyi DataFrame'e ekle
    new_data = {
        'text': test_input,
        'entity': entity,
        'sentiment': sentiment,
        'konu': konu,
        'severity': severity,
        'bilet': multilabel['bilet'],
        'musteri_hizmetleri': multilabel['musteri_hizmetleri'],
        'odeme': multilabel['odeme'],
        'uygulama': multilabel['uygulama'],
        'passolig': multilabel['passolig'],
        'passolig kart': multilabel['passolig kart'],
        'diger': multilabel['diger'],
        'aksiyon': model_predictions['Severity']['action_status'],
        'llm_response': llm_input
    }

    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv('user_input.csv', index=False)

# Örnek kullanım
df = pd.DataFrame()  # Boş bir DataFrame başlatın
test_input = "Ödeme yaparken hata aldım ve müşteri hizmetleri ilgilenmedi."
model_predictions = {
    'Entity': 'odeme',
    'Konu': 'odeme',
    'Sentiment': {'label': 'olumsuz'},
    'Severity': {'severity_label': 2, 'action_status': 1},
    'Multilabel': {
        'bilet': 0,
        'musteri_hizmetleri': 1,
        'odeme': 1,
        'uygulama': 0,
        'passolig': 0,
        'passolig kart': 0,
        'diger': 0
    }
}

# Fonksiyonu çalıştır
process_data(df, test_input, model_predictions)
