# categorize.py
# Bu dosya şikayetleri kategorilere ayırma algoritmalarını içerir.
# Şikayetlerin veya metinlerin konu başlıklarına göre kategorize edilmesini sağlar.

def categorize_complaint(text):
    """
    Şikayeti kategorilere ayırır.
    :param text: Şikayet metni
    :return: Şikayet kategorisi
    """
    # Basit bir örnek kategorizasyon
    if "ödem" in text:
        return "Ödeme"
    elif "servis" in text:
        return "Servis"
    else:
        return "Diğer"

if __name__ == "__main__":
    text = "Ürün servisi çok kötü."
    category = categorize_complaint(text)
    print(f"Şikayet Kategorisi: {category}")
