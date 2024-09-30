import pandas as pd
import joblib  # pickle yerine joblib kullanıyoruz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# 'konu' değerleri
konu_labels = [
    'cagri merkezi yetkinlik', 'diger', 'genel', 'odeme', 'uygulama',
    'iptal', 'degisiklik', 'uyelik', 'iade', 'transfer', 'fatura'
]

def train_and_save_model(data_path, model_output_path):
    # Veriyi yükle
    data = pd.read_csv(data_path)
    
    # Metin ve konu etiketlerini ayır
    X = data['text']  # Metin sütunu
    y = data['konu']  # Konu etiketleri sütunu
    
    # Etiketleri sayısal değerlere dönüştür
    y = pd.Categorical(y, categories=konu_labels).codes
    
    # Veriyi eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SVM modelini ve TF-IDF vektörleştiricisini birleştirin
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Modeli dosyaya kaydedin
    joblib.dump(model, model_output_path)  # joblib kullanarak kaydediyoruz
    print(f"Model kaydedildi: {model_output_path}")

def test_model(model_path, sentence):
    model = joblib.load(model_path)  # joblib kullanarak modeli yüklüyoruz
    
    # Cümleyi tahmin et
    predicted_label = model.predict([sentence])
    print(f"Tahmin edilen konu: {konu_labels[predicted_label[0]]}")

if __name__ == "__main__":
    # Veri ve model dosya yolları
    data_path = 'data/processed/cleaned_df.csv'  # Veri setinin yolu
    model_output_path = 'data/models/konu_model.joblib'  # Eğitilen modelin kaydedileceği yol
    
    # Modeli eğit ve kaydet
    train_and_save_model(data_path, model_output_path)
    
    # Modeli test etme
    while True:
        test_input = input("Bir metin girin (çıkmak için 'q' yazın): ")
        if test_input.lower() == 'q':
            break
        test_model(model_output_path, test_input)
