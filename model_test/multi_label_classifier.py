import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def train_and_save_model(data_path, model_output_path):
    # Veriyi yükleme
    df = pd.read_csv(data_path)

    # Özellikler (X) ve hedef etiketler (y)
    X = df['text']
    y = df[['bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig kart', 'diger']]

    # Text verisini TF-IDF ile vektörize etme
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Veriyi eğitim ve test setlerine bölme
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Modeli tanımlama ve en iyi parametrelerle eğitme
    model = OneVsRestClassifier(RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    ))

    # Modeli eğitme
    model.fit(X_train, y_train)

    # Eğitim ve test skorlarını yazdırma
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")
    print("-" * 40)

    # Modeli kaydetme
    joblib.dump((model, tfidf), model_output_path)
    print(f"Model başarıyla kaydedildi: {model_output_path}")

    # Test seti üzerinde performansı değerlendirme
    y_pred = model.predict(X_test)
    print("\nModel Performansı:")
    print(classification_report(y_test, y_pred, target_names=y.columns))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

def test_model(model_output_path, input_text):
    # Modeli yükleme
    model, tfidf = joblib.load(model_output_path)
    
    # Giriş metnini TF-IDF ile vektörize etme
    input_tfidf = tfidf.transform([input_text])
    
    # Tahmin yapma
    prediction = model.predict(input_tfidf)
    
    # Tahmin edilen etiketler
    predicted_labels = dict(zip(['bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig kart', 'diger'], prediction[0]))
    
    # Sonuçları göster
    print("Tahmin Edilen Etiketler:")
    for label, value in predicted_labels.items():
        print(f"{label}: {value}")

    # İlgili sütunların 0 veya 1 ile işaretlenmesi
    result = {label: 1 if value == 1 else 0 for label, value in predicted_labels.items()}
    print("Sonuçlar:")
    for label, value in result.items():
        print(f"{label}: {value}")

if __name__ == "__main__":
    # Veri ve model dosya yolları
    data_path = 'data/processed/cleaned_df.csv'  # Veri setinin yolu
    model_output_path = 'data/models/multilabel/multilabelclassifier.pkl'  # Eğitilen modelin kaydedileceği yol
    
    # Modeli eğit ve kaydet
    train_and_save_model(data_path, model_output_path)
    
    # Modeli test etme
    test_input = input("Bir metin girin: ")  # Kullanıcıdan metin girişi
    test_model(model_output_path, test_input)
