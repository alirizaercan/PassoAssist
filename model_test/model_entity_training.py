import re
import spacy
import pickle

# 'tr_core_news_trf' modelini yükleyin
nlp = spacy.load("tr_core_news_trf")

# Passo, Passolig, Passolig Kart gibi entity'leri listeye ekleyin
entities = ["passo", "passolig", "passolig kart"]

# Yanlış yazımların düzeltilmesi için bir sözlük
corrected_spellings = {
    "passolg": "passolig",
    "passolig krt": "passolig kart",
    # İhtiyaç duyulursa daha fazla yanlış yazım ekleyin
}

def correct_spelling(sentence):
    # Cümledeki kelimeleri kontrol edip yanlış yazımları düzelt
    words = sentence.split()
    corrected_words = []
    for word in words:
        # Eğer kelime bir yanlış yazım ise düzeltilmiş halini ekle
        corrected_word = corrected_spellings.get(word.lower(), word)
        corrected_words.append(corrected_word)
    return ' '.join(corrected_words)

def find_entities(sentence, entity_list):
    # Yazım hatalarını düzelt
    corrected_sentence = correct_spelling(sentence)
    found_entities = []
    for entity in entity_list:
        # Entity'yi case-insensitive olarak bulmak için regex kullanıyoruz
        # Eklere de dikkat edecek şekilde güncellenmiş regex deseni
        pattern = rf'\b{entity}(?:\w+)?\b'
        if re.search(pattern, corrected_sentence, re.IGNORECASE):
            found_entities.append(entity)
    return found_entities

def process_sentence(sentence, entity_list):
    found_entities = find_entities(sentence, entity_list)
    if found_entities:
        # Bulunan entity'leri noktalı virgül ile birleştir
        return "; ".join(found_entities)
    return "No entity found."

def train_and_save_model(data_path, model_output_path):
    # Basit bir model eğitimi örneği (burada dummy olarak gösteriliyor)
    print(f"Eğitim verisi: {data_path}")
    model = {"dummy_model": "Bu bir örnek modeldir"}
    
    # Modeli dosyaya kaydedin
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model kaydedildi: {model_output_path}")

def test_model(model_path, sentence):
    # Kaydedilen modeli yükleyin
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model yüklendi: {model_path}")
    
    # Giriş cümlesindeki entity'leri bulun ve yazdırın
    entities_found = process_sentence(sentence, entities)
    print(f"Found entities: {entities_found}")

if __name__ == "__main__":
    # Veri ve model dosya yolları
    data_path = 'data/processed/cleaned_df.csv'  # Veri setinin yolu
    model_output_path = 'data/models/entity_model.pkl'  # Eğitilen modelin kaydedileceği yol
    
    # Modeli eğit ve kaydet
    train_and_save_model(data_path, model_output_path)
    
    # Modeli test etme
    while True:
        test_input = input("Bir metin girin (çıkmak için 'q' yazın): ")
        if test_input.lower() == 'q':
            break
        test_model(model_output_path, test_input)
