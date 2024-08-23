# PassoAssist

## Türkçe [TR]

### Proje Açıklaması

**PassoAssist**: Bu proje, şikayet yönetimi ve çözümü için akıllı bir sistem geliştirmeyi hedefler. Proje, veri bilimi ve doğal dil işleme (NLP) tekniklerini kullanarak kullanıcı geri bildirimlerini analiz eder ve anlamlı sonuçlar üretir. Sistem, web scraping yöntemleriyle veri toplar, verileri işleyip analiz eder ve sonuçları kullanıcı dostu bir arayüzde sunar.

### Veri Bilimi Açısından Önemi

**PassoAssist**, veri bilimi açısından büyük bir önem taşır çünkü:

1. **Veri Toplama ve Ön İşleme**: Web scraping yöntemleri ile toplanan büyük veri kümesi üzerinde veri ön işleme adımları uygulanır. Bu süreç, verinin kalitesini artırarak doğru analizler yapılmasını sağlar.
2. **Sentiment Analizi**: Doğal dil işleme teknikleri kullanılarak metinlerin sentiment analizi yapılır. Bu analiz, şikayetlerin duygusal tonunu anlamak ve kullanıcılara uygun çözümler sunmak için önemlidir.
3. **Kategorilendirme**: Şikayetlerin türlerine göre kategorilere ayrılması, veri yönetimini ve çözüm önerilerini daha etkili hale getirir.
4. **Gerçek Zamanlı Analiz**: Web arayüzü üzerinden kullanıcıların veri talep edebilmesi ve sonuçları anlık olarak görmesi sağlanır. Bu, kullanıcılara hızlı geri dönüşler sunar ve veri bilimi uygulamalarının etkinliğini artırır.

### Nasıl Çalışır?

1. **Veri Toplama**: `scraping.py` dosyasında web scraping işlemleri gerçekleştirilir ve veriler `data/raw/` klasörüne kaydedilir.
2. **Veri Ön İşleme**: `preprocessing.py` dosyasında veriler temizlenir ve işlenir. İşlenmiş veriler `data/processed/` klasörüne kaydedilir.
3. **Model Eğitimi**: `train_model.py` dosyasında NLP modeli eğitilir ve model `data/models/` klasörüne kaydedilir.
4. **Tahmin ve Kategorilendirme**: `predict.py` ve `categorize.py` dosyalarında tahmin ve kategorilendirme işlemleri yapılır.
5. **Web Uygulaması**: `app.py` dosyasında Flask/Django ile web uygulaması başlatılır ve API endpoint'leri tanımlanır.
6. **Web Arayüzü**: `index.html`, `styles.css`, ve `app.js` dosyalarında kullanıcı arayüzü oluşturulur ve dinamik özellikler eklenir.

### Kurulum ve Çalıştırma

1. Gerekli paketleri yükleyin: `pip install -r requirements.txt`
2. Web scraping işlemini başlatın: `python scripts/scraping.py`
3. Veri ön işleme adımlarını uygulayın: `python scripts/preprocessing.py`
4. Modeli eğitin: `python scripts/train_model.py`
5. Web uygulamasını başlatın: `python src/app.py`

## English [EN]

### Project Description

**PassoAssist**: This project aims to develop an intelligent system for complaint management and resolution. Using data science and natural language processing (NLP) techniques, the system analyzes user feedback and generates meaningful results. The system collects data through web scraping methods, processes and analyzes the data, and presents the results through a user-friendly interface.

### Importance from a Data Science Perspective

**PassoAssist** is significant from a data science perspective because:

1. **Data Collection and Preprocessing**: Large datasets collected through web scraping are preprocessed to enhance data quality, enabling accurate analyses.
2. **Sentiment Analysis**: NLP techniques are used to perform sentiment analysis on texts. This analysis is crucial for understanding the emotional tone of complaints and providing appropriate solutions.
3. **Categorization**: Categorizing complaints by their types makes data management and solution proposals more effective.
4. **Real-Time Analysis**: Users can request data and view results in real-time through the web interface. This provides quick feedback and enhances the effectiveness of data science applications.

### How It Works

1. **Data Collection**: Web scraping operations are carried out in the `scraping.py` file, and data is saved in the `data/raw/` directory.
2. **Data Preprocessing**: Data is cleaned and processed in the `preprocessing.py` file, and processed data is saved in the `data/processed/` directory.
3. **Model Training**: An NLP model is trained in the `train_model.py` file, and the model is saved in the `data/models/` directory.
4. **Prediction and Categorization**: Prediction and categorization are performed in the `predict.py` and `categorize.py` files.
5. **Web Application**: The web application is started using Flask/Django in the `app.py` file, and API endpoints are defined.
6. **Web Interface**: The user interface is created and dynamic features are added in the `index.html`, `styles.css`, and `app.js` files.

### Installation and Running

1. Install required packages: `pip install -r requirements.txt`
2. Start web scraping: `python scripts/scraping.py`
3. Apply data preprocessing steps: `python scripts/preprocessing.py`
4. Train the model: `python scripts/train_model.py`
5. Start the web application: `python src/app.py`
