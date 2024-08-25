import pandas as pd
import random
import csv

# Mevcut verileri yükleyin
data_path = 'data/raw/raw_data_passo.csv'
df = pd.read_csv(data_path)

# Sentetik veriler üretilecek kategoriler
categories = ['bilet', 'müşteri hizmetleri', 'ödeme', 'uygulama', 'passolig', 'passolig kart']

# Karmaşık olumlu ve nötr yorum şablonları (olumlu ve nötr kelimeler içeren)
mixed_sentiments_templates = [
    "Passo'nun {} işlemleri hızlı ve sorunsuzdu, ama bu kadar basit bir şeyi bu kadar uzun süre beklemek sinir bozucuydu.",
    "Passolig kartımla {} işlemi sonunda halledebildim, ama müşteri hizmetlerinin ilgisi yetersizdi.",
    "{} işlemleri sorunsuz gitti, ama uygulamanın arayüzü biraz daha iyi olabilirdi.",
    "Passo ile {} işlemleri sorunsuz gerçekleşti, fakat uygulamanın performansı biraz yavaş olabiliyor.",
    "Uygulama üzerinden {} yapmak oldukça kolaydı, ama bazı özellikler geliştirilmeli.",
]

# Pozitif, nötr cümleler ekleyelim
positive_templates = [
    "Passo'nun {} ile ilgili hizmeti gerçekten mükemmeldi, hiçbir sorun yaşamadım.",
    "Passolig kartımla {} işlemi inanılmaz derecede kolaydı, harika bir deneyim yaşadım.",
    "{} işlemlerim Passo sayesinde sorunsuz bir şekilde tamamlandı, teşekkürler.",
    "Passo uygulaması üzerinden {} ile ilgili herhangi bir aksaklık olmadı, işini iyi yapıyor.",
    "Müşteri hizmetleri {} konusunda çok hızlı ve etkili davrandı, takdir ediyorum.",
    "{} işlemlerim çok hızlı gerçekleşti, gerçekten memnunum.",
    "{} ile ilgili hiçbir problem yaşamadım, tam anlamıyla mükemmel hizmet!",
    "Passo sayesinde {} ile ilgili her şeyi çok hızlı ve rahat bir şekilde halledebildim.",
    "{} konusunda Passo her zaman yanımda oldu, en ufak bir aksaklık bile yaşamadım.",
    "Passo ile {} sürecim kusursuzdu, her şey beklentilerimin ötesindeydi.",
    "Uygulama üzerinden {} yaparken hiçbir sıkıntı yaşamadım, gayet kullanıcı dostu.",
    "Passo ekibi {} konusunda profesyonel bir destek sağladı, çok memnun kaldım.",
    "{} işlemlerim sırasında müşteri hizmetleri çok ilgiliydi, tüm sorularımı çözdüler.",
    "Passolig kartım ile {} işlemi son derece hızlı ve sorunsuz bir şekilde halledildi.",
    "Passo uygulamasıyla {} yapmak çok pratik ve zaman kazandırıcı oldu.",
    "{} konusunda Passo'nun sunduğu hizmet beni gerçekten etkiledi, her şey harikaydı.",
    "{} işlemlerim Passo sayesinde anında çözüldü, mükemmel bir sistem!",
    "{} işlemi sırasında hiç beklemediğim kadar iyi bir deneyim yaşadım, Passo'ya teşekkür ederim.",
    "Passo ile {} her zamanki gibi hızlı ve güvenilir bir şekilde halledildi, mükemmel!"
]

neutral_templates = [
    "{} ile ilgili hizmet Passo tarafından normal düzeyde sunuldu, ekstra bir durum yok.",
    "Passolig kartımla ilgili {} işlemi herhangi bir aksaklık olmadan gerçekleşti.",
    "{} işlemi hakkında ne olumlu ne de olumsuz bir deneyimim var.",
    "{} üzerinden birkaç işlem yaptım, genel olarak standart bir hizmetti.",
    "{} konusunda Passo'nun sunduğu hizmet oldukça sıradandı.",
    "{} hizmeti yeterliydi, herhangi bir sorun yaşamadım ama ekstra da bir şey sunmadı.",
    "{} işlemi gayet normal bir şekilde tamamlandı, herhangi bir sorun yaşamadım.",
    "{} sürecinde ne büyük bir problem yaşadım ne de çok olumlu bir şey fark ettim.",
    "Passo üzerinden {} yaparken herhangi bir aksaklık yaşamadım ama çok da etkilenmedim.",
    "{} işlemi standart bir hızda gerçekleşti, çok sıra dışı bir şey yaşamadım.",
    "{} konusunda Passo'nun hizmeti bana sıradan geldi, ne çok iyi ne de çok kötü.",
    "{} işlemlerim herhangi bir problem olmadan tamamlandı, gayet sıradandı.",
    "{} konusunda Passo'nun sunduğu hizmet, genel olarak ortalama düzeydeydi.",
    "{} ile ilgili herhangi bir sorun yaşamadım, standart bir deneyimdi.",
    "{} konusunda Passo'nun hizmeti iyiydi ama olağanüstü bir şey söyleyemem.",
    "Passo ile {} işlemlerimi tamamladım, ama hizmet sıradandı, fazla bir şey beklememiştim.",
    "{} işlemi normal bir hızda gerçekleşti, ne iyi ne de kötü bir deneyim oldu.",
    "{} hizmeti hakkında net bir görüşüm yok, herhangi bir olumsuzlukla karşılaşmadım."
]
# Rastgele varyasyonlar üretmek için ekleyebileceğimiz kelimeler
random_phrases = [
    "gerçekten", "çok", "son derece", "beklediğim gibi", "sorunsuzca", "kolaylıkla"
]

# 500 adet sahte yorum oluştur
num_samples = 500
synthetic_data = []

for _ in range(num_samples):
    category = random.choice(categories)
    
    # Karmaşık, olumlu veya nötr yorumları rastgele seç
    if random.random() < 0.4:
        template = random.choice(mixed_sentiments_templates)  # Karmaşık yorumlar
    elif random.random() < 0.7:
        template = random.choice(positive_templates)  # Olumlu
    else:
        template = random.choice(neutral_templates)  # Nötr
    
    # Cümlede rastgele bir yere fazladan bir ifade ekle
    random_phrase = random.choice(random_phrases)
    comment = template.format(category)
    
    # Şanslı bir şekilde, cümlenin bir yerine fazladan ifade ekleyelim
    if random.random() > 0.5:
        words = comment.split()
        insert_idx = random.randint(1, len(words) - 1)
        words.insert(insert_idx, random_phrase)
        comment = ' '.join(words)
    
    synthetic_data.append(comment)

# Mevcut CSV'deki son satırdan itibaren ekleme yapın
starting_index = len(df)  # 2686. satırdan başlayacak şekilde otomatik hesaplama

# Yeni verileri dataframe olarak ekleyin
synthetic_df = pd.DataFrame({'text': synthetic_data})
df = pd.concat([df, synthetic_df], ignore_index=True)

# CSV'ye verileri kaydedin
df.to_csv(data_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"{num_samples} sentetik olumlu ve nötr yorum başarıyla {data_path} dosyasına eklendi.")
