import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_and_save_model(data_path, model_output_path):
    # Veriyi yükleme
    df = pd.read_csv(data_path)

    # Özellikler (X) ve hedef etiketler (y)
    X = df['text']
    y = df['severity']  # Burada sadece severity hedef alındı

    # Text verisini TF-IDF ile vektörize etme
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Veriyi eğitim ve test setlerine bölme
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Modeli tanımlama ve eğitme
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Modeli kaydetme
    joblib.dump((model, tfidf), model_output_path)

    # Test seti üzerinde performansı değerlendirme
    y_pred = model.predict(X_test)
    print("\nModel Performansı:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

def test_model(model_output_path, input_text):
    # Modeli yükleme
    model, tfidf = joblib.load(model_output_path)
    
    # Giriş metnini TF-IDF ile vektörize etme
    input_tfidf = tfidf.transform([input_text])
    
    # Tahmin yapma
    severity_prediction = model.predict(input_tfidf)
    
    # Severity tahminini yazdırma
    print(f"Tahmin Edilen Severity: {severity_prediction[0]}")

    # Aksiyon durumu belirleme
    if severity_prediction[0] == 2:
        action_status = 1  # Çok acil
        action_message = "Acil Harekete Geçin!"
    elif severity_prediction[0] == 1:
        action_status = 1  # Acil değil ama önemli
        action_message = "Aksiyon Almanız Önerilir."
    else:
        action_status = 0  # Hiç aksiyon gerekmiyor
        action_message = "Harekete Geçmeye Gerek Yok."

    print(f"Aksiyon Durumu: {action_status} ({action_message})")

if __name__ == "__main__":
    # Veri ve model dosya yolları
    data_path = 'data/processed/cleaned_df.csv'  # Veri setinin yolu
    model_output_path = 'data/models/severity_classifier.joblib'  # Eğitilen modelin kaydedileceği yol
    
    # Modeli eğit ve kaydet
    train_and_save_model(data_path, model_output_path)
    
    # Modeli test etme
    while True:
        test_input = input("Bir metin girin (çıkmak için 'q' yazın): ")
        if test_input.lower() == 'q':
            break
        test_model(model_output_path, test_input)
