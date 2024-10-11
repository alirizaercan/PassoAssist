from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Model ve tokenizer'ları yükle
gpt2_model_name = "cenkersisman/gpt2-turkish-128-token"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

turna_tokenizer = AutoTokenizer.from_pretrained("boun-tabi-LMG/TURNA")
turna_model = AutoModelForSeq2SeqLM.from_pretrained("boun-tabi-LMG/TURNA")

# Kullanıcıdan metin girişi ve data bilgilerini alma
user_input_text = input("Lütfen şikayet metninizi girin: ")

data = {
    'bilet': int(input("Bilet durumu (0 veya 1): ")),
    'musteri_hizmetleri': int(input("Müşteri hizmetleri durumu (0 veya 1): ")),
    'odeme': int(input("Ödeme durumu (0 veya 1): ")),
    'uygulama': int(input("Uygulama durumu (0 veya 1): ")),
    'passolig': int(input("Passolig durumu (0 veya 1): ")),
    'passolig kart': int(input("Passolig kart durumu (0 veya 1): ")),
    'diger': int(input("Diğer durumu (0 veya 1): ")),
    'konu': input("Konu (örn. 'odeme', 'uygulama', 'passolig' vb.): "),
    'entity': input("Entity (örn. 'passo', 'passolig', 'passolig kart' vb.): "),
    'sentiment': input("Duygu durumu (olumsuz, nötr, olumlu): "),
    'severity': int(input("Ciddiyet durumu (0, 1 veya 2): ")),
    'aksiyon': int(input("Aksiyon durumu (0 veya 1): "))
}

def generate_response(data, user_input):
    response_parts = []

    # Ciddiyet
    if data['severity'] == 2:
        response_parts.append("Bu konu acil bir durum olarak değerlendirilmiştir ve öncelikli olarak ele alınacaktır.")
    elif data['severity'] == 1:
        response_parts.append("Bu konunun aciliyeti vardır ve en kısa zamanda ilgilenilecektir.")
    elif data['severity'] == 0:
        response_parts.append("Bu konu ilgili ekip tarafından gözden geçirilecektir.")

    # Duygu durumu
    if data['sentiment'] == 'olumsuz':
        response_parts.append("Yaşadığınız sorun için üzgünüz. Önerilerimiz: ")
        if data['odeme'] == 1:
            response_parts.append("Ödeme yöntemlerinizi gözden geçirmenizi öneririz.")
        if data['uygulama'] == 1:
            response_parts.append("Uygulamayı güncelleyerek tekrar deneyin.")
    elif data['sentiment'] == 'notr':
        response_parts.append("Önerimiz: Ödeme işlemlerinizi dikkatlice kontrol edin.")
    elif data['sentiment'] == 'olumlu':
        response_parts.append("Teşekkür ederiz! Size yardımcı olmaktan memnuniyet duyarız.")

    # Konu
    if data['konu']:
        response_parts.append(f"Konu: {data['konu']} ile ilgili gerekli adımları atacağız.")
        
    # Entity
    if data['entity'] in ["passo", "passolig", "passolig kart"]:
        response_parts.append(f"{data['entity'].capitalize()} ile ilgili gerekli adımları atacağız.")

    # Aksiyon durumu
    if data['aksiyon'] == 1:
        response_parts.append("Harekete geçilecektir, lütfen bekleyin.")
    else:
        response_parts.append("Bu aşamada ek bir aksiyon alınmayacaktır.")

    return " ".join(response_parts)

# Yanıt oluşturma
response = generate_response(data, user_input_text)
print("Kurumsal Yanıt:", response)

# LLM ile yanıt oluşturma
generator = pipeline("text2text-generation", model=turna_model, tokenizer=turna_tokenizer)

def generate_response_with_llm(data, user_input):
    prompt = f"Kullanıcı şikayeti: '{user_input}'. {generate_response(data, user_input)}. Bununla ilgili bir yanıt oluşturun."
    
    return generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

llm_response = generate_response_with_llm(data, user_input_text)
print("LLM Yanıtı:", llm_response)

# GPT-2 ile yanıt oluşturma
gpt2_prompt = f"Kullanıcı şikayeti: '{user_input_text}'. Yanıt olarak şunları önerin: " + response
gpt2_input_ids = gpt2_tokenizer.encode(gpt2_prompt, return_tensors="pt")

# Yeni token sayısını belirleme
max_new_tokens = 50

# Modeli çalıştırma
gpt2_output = gpt2_model.generate(gpt2_input_ids, max_new_tokens=max_new_tokens, pad_token_id=gpt2_tokenizer.eos_token_id)
gpt2_generated_text = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

print("GPT-2 Yanıtı:", gpt2_generated_text)
