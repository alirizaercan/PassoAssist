import pandas as pd
import joblib

def test_model(model_output_path, input_text):
    """Modeli yükleyerek verilen metin için tahmin yapar."""
    # Modeli yükleme
    model, tfidf = joblib.load(model_output_path)
    
    # Giriş metnini TF-IDF ile vektörize etme
    input_tfidf = tfidf.transform([input_text])
    
    # Tahmin yapma
    prediction = model.predict(input_tfidf)
    
    # Tahmin edilen etiketler
    predicted_labels = dict(zip(['bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig kart', 'diger'], prediction[0]))
    
    # İlgili sütunların 0 veya 1 ile işaretlenmesi
    result = {label: 1 if value == 1 else 0 for label, value in predicted_labels.items()}
    
    return result

def main():
    model_output_path = 'data/models/multilabel/multilabelclassifier.pkl'  # Eğitilen modelin kaydedileceği yol
    print("Model yükleniyor...")
    
    while True:
        text = input("Bir metin girin (çıkmak için 'exit' yazın): ")
        if text.lower() == 'exit':
            print("Programdan çıkılıyor.")
            break
        
        results = test_model(model_output_path, text)
        print("Sonuçlar:")
        for label, value in results.items():
            print(f"{label}: {value}")

if __name__ == "__main__":
    main()
