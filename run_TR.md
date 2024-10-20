
# PassoAssist
# PassoAssist Projesi Kurulum Talimatları

Bu dosya, PassoAssist chatbot projesinin adım adım nasıl kurulacağını ve çalıştırılacağını anlatmaktadır.

---

### 1. Gereksinimleri Yükleme

Projeyi bilgisayarınıza indirdikten sonra, ihtiyaç duyulan kütüphaneleri yüklemek için `requirements.txt` dosyasını kullanın:

```bash
pip install -r requirements.txt
```

### 2. Verileri Toplama
Veriler ŞikayetVar sitesinden toplanmaktadır. Bu verileri scraping yöntemi ile almak için:

```bash
cd scripts
python scraping.py
```

### 3. Sentetik Veri Üretimi (Opsiyonel)
Sentetik veri üretmek isterseniz, şu komutu çalıştırın:

```bash
python generate_synthetic_data.py
```

### 4. Veri Temizleme
Veri temizleme işlemi için şu komutları kullanın:

```bash
cd scripts/model_training_scripts
python text_cleaning.py
```

### 5. Model Eğitimi
Eğitilen modeller için:

```bash
cd src/nlp
python train_model.py
```

### 6. Tahmin Yapma
Temizlenmiş veriler üzerinde tahmin yapmak için:

```bash
python predict.py
```

### 7. Chatbotu Çalıştırma
Yerel ortamda chatbotu çalıştırmak için:

```bash
python app.py
```

### Sorunlar ve Katkıda Bulunma
Proje ile ilgili sorunlarınızı Issues kısmında belirtebilir veya LinkedIn üzerinden iletişime geçebilirsiniz.




