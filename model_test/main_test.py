from flask import Flask, render_template, request, jsonify
from cleaning_service import process_user_input
from response_with_all_models import predict_all_models, generate_professional_response
import numpy as np

app = Flask(__name__)

# CSS versiyonu
CSS_VERSION = "1.0.1"  # Bu versiyonu her değişiklikte güncelleyin

# Ana sayfa için rota
@app.route('/')
def home():
    return render_template('passo.html', version=CSS_VERSION)

@app.route('/chatbot', methods=['POST'])
def clean_text():
    try:
        data = request.get_json()
        user_input = data.get('text', '')

        if not user_input:
            return jsonify({'error': 'Metin verisi gerekli!'}), 400

        # Kullanıcı girişini temizle
        cleaned_text = process_user_input(user_input, to_english=True)

        # Önceden tanımlanmış yanıtlar
        predefined_responses = {
            'greeting': "Merhaba! Size nasıl yardımcı olabilirim?",
            'feeling': "Teşekkür ederim, iyiyim. Sizin için ne yapabilirim?",
            'identity': "Ben PassoAssist, Ali Rıza Ercan tarafından geliştirildim. Size nasıl yardımcı olabilirim?",
            'help_request': "Tabii ki! Size nasıl yardımcı olabilirim? Sorununuzu bana anlatabilirsiniz.",
            'info_request': "Hangi konu hakkında bilgi almak istersiniz? Size en iyi şekilde yardımcı olabilirim.",
            'thanks': "Rica ederim! Size her zaman yardımcı olmaktan mutluluk duyarım.",
            'positive_feedback': "Olumlu geri bildiriminiz için teşekkür ederim! Size daha iyi hizmet verebilmek adına çalışmalarımıza devam edeceğiz.",
            'negative_feedback': "Yaşamış olduğunuz aksaklık için özür dilerim. Sorununuzu çözmek için buradayım."
        }

        # Önceden tanımlanmış yanıtları kontrol et
        if any(greeting in user_input.lower() for greeting in [
            'merhaba', 'selam', 'hi', 'hey', 'selamlar', 'merhabalar',
            'merhabe', 'selamlasma', 'merhabalarim', 'merhaba arkadas',
            'selam dostum', 'merhaba arkadasim']):
            response_message = predefined_responses['greeting']
        elif any(feeling in user_input.lower() for feeling in [
            'nasilsin', 'iyi misin', 'nasilsiniz', 'iyi misiniz',
            'nasil gidiyor', 'keyfiniz nasıl', 'nasılsınız']):
            response_message = predefined_responses['feeling']
        elif any(identity in user_input.lower() for identity in [
            'adin ne', 'sen kimsin', 'adiniz ne', 'siz kimsiniz',
            'sen nesin', 'adınız kim', 'kimdirsiniz']):
            response_message = predefined_responses['identity']
        elif any(help_request in user_input.lower() for help_request in [
            'yardim et', 'bana yardim et', 'yardima ihtiyacim var', 'yardim edin',
            'yardim lütfen', 'yardim rica ederim', 'yardim bulmak istiyorum']):
            response_message = predefined_responses['help_request']
        elif any(info_request in user_input.lower() for info_request in [
            'bilgi ver', 'bilgi alabilir miyim', 'bilgi verir misiniz', 'bana bilgi ver',
            'bilgi lütfen', 'bilgi almak istiyorum', 'bilgi isterim']):
            response_message = predefined_responses['info_request']
        elif any(thanks in user_input.lower() for thanks in [
            'tesekkur ederim', 'sag ol', 'teşekkür ederim', 'sağ ol',
            'teşekkürler', 'sagolsun', 'teşekkürler dostum']):
            response_message = predefined_responses['thanks']
        else:
            # Eğer önceden tanımlanmış bir yanıt yoksa model tahminlerini al
            model_outputs = predict_all_models(cleaned_text)

            # Model çıktılarında sayıların tam sayıya dönüştürülmesi
            for key in model_outputs:
                if isinstance(model_outputs[key], dict):
                    for sub_key in model_outputs[key]:
                        if isinstance(model_outputs[key][sub_key], (int, np.int32, np.int64)):
                            model_outputs[key][sub_key] = int(model_outputs[key][sub_key])
                elif isinstance(model_outputs[key], (int, np.int32, np.int64)):
                    model_outputs[key] = int(model_outputs[key])

            # Profesyonel yanıt oluştur
            response_message = generate_professional_response(model_outputs)

        # JSON formatında yanıt döndür
        response = jsonify({
            'text': user_input,
            'cleaned_text': cleaned_text,
            'model_outputs': model_outputs if 'model_outputs' in locals() else {},  # Eğer model_outputs yoksa boş bir dict gönder
            'response_message': response_message
        })

        # Önbellek kontrol başlıklarını ekle
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        return response, 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
