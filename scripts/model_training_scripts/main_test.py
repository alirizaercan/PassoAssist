from flask import Flask, render_template, request, jsonify
from cleaning_service import process_user_input
from response_with_all_models import predict_all_models, generate_professional_response
import pandas as pd
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
            'negative_feedback': "Yaşamış olduğunuz aksaklık için özür dilerim. Sorununuzu çözmek için buradayım.",
            'issue': "Yaşadığınız sorunu daha iyi anlayabilmemiz için lütfen gerekli detayları paylaşın. En kısa sürede çözüm sunabilmek adına size yardımcı olmaktan memnuniyet duyarız."
        }

        response_message = None
        model_outputs = {}

        # Önceden tanımlanmış yanıtları kontrol et
        if any(greeting in user_input.lower() for greeting in [
            'merhaba', 'selam', 'hi', 'hey', 'selamlar', 'merhabalar',
            'merhabe', 'selamlasma', 'merhabalarim', 'merhaba arkadas',
            'selam dostum', 'merhaba arkadasim']):
            response_message = predefined_responses['greeting']
        elif any(feeling in user_input.lower() for feeling in [
            'nasilsin', 'iyi misin', 'nasilsiniz', 'iyi misiniz',
            'nasil gidiyor', 'keyfiniz nasil', 'nasilsiniz']):
            response_message = predefined_responses['feeling']
        elif any(identity in user_input.lower() for identity in [
            'adin ne', 'sen kimsin', 'adiniz ne', 'siz kimsiniz',
            'sen nesin', 'adiniz kim', 'kimdirsiniz']):
            response_message = predefined_responses['identity']
        elif any(help_request in user_input.lower() for help_request in [
            'yardim et', 'bana yardim et', 'yardima ihtiyacim var', 'yardim edin',
            'yardim lutfen', 'yardim rica ederim', 'yardim bulmak istiyorum']):
            response_message = predefined_responses['help_request']
        elif any(info_request in user_input.lower() for info_request in [
            'bilgi ver', 'bilgi alabilir miyim', 'bilgi verir misiniz', 'bana bilgi ver',
            'bilgi lutfen', 'bilgi almak istiyorum', 'bilgi isterim']):
            response_message = predefined_responses['info_request']
        elif any(thanks in user_input.lower() for thanks in [
            'tesekkur ederim', 'sag ol', 'tesekkur ederim', 'sag ol',
            'tesekkurler', 'sagolsun', 'tesekkurler dostum']):
            response_message = predefined_responses['thanks']
        elif any(issue in user_input.lower() for issue in [
            'sorunum var', 'problemim var', 'bir sorun yasiyorum', 'bir problemim var', 
            'sikinti var', 'sorunla karsilastim', 'problemle karsilastim']):
            response_message = predefined_responses['issue']
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


        # JSON formatında yanıt döndürme
        response = jsonify({
            'text': user_input,
            'cleaned_text': cleaned_text,
            'model_outputs': model_outputs,
            'response_message': response_message
        })
        
        print(model_outputs)
        
        # Veriyi CSV'ye kaydetme
        new_row = pd.DataFrame([{
            'text': user_input,
            'entity': model_outputs['Entity'] if 'Entity' in model_outputs else '',
            'sentiment': model_outputs['Sentiment']['label'] if 'Sentiment' in model_outputs else '',
            'konu': model_outputs['Konu'] if 'Konu' in model_outputs else '',
            'severity': model_outputs['Severity']['severity_label'] if 'Severity' in model_outputs else '',
            'bilet': model_outputs['Multilabel']['bilet'] if 'Multilabel' in model_outputs else '',
            'musteri_hizmetleri': model_outputs['Multilabel']['musteri_hizmetleri'] if 'Multilabel' in model_outputs else '',
            'odeme': model_outputs['Multilabel']['odeme'] if 'Multilabel' in model_outputs else '',
            'uygulama': model_outputs['Multilabel']['uygulama'] if 'Multilabel' in model_outputs else '',
            'passolig': model_outputs['Multilabel']['passolig'] if 'Multilabel' in model_outputs else '',
            'passolig kart': model_outputs['Multilabel']['passolig kart'] if 'Multilabel' in model_outputs else '',
            'diger': model_outputs['Multilabel']['diger'] if 'Multilabel' in model_outputs else '',
            'aksiyon': model_outputs['Severity']['action_status'] if 'Severity' in model_outputs else '',
            'sentiment_confidence': model_outputs['Sentiment']['confidence'] if 'Sentiment' in model_outputs else '',
            'response': response_message
        }])

        # CSV dosyasını güncelleme
        try:
            df = pd.read_csv('data/test/test_input.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['text', 'entity', 'sentiment', 'konu', 
                                       'severity', 'bilet', 'musteri_hizmetleri',
                                       'odeme', 'uygulama', 'passolig', 
                                       'passolig kart', 'diger', 'aksiyon',
                                       'sentiment_confidence',  'response'])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv('data/test/test_input.csv', index=False)
        print("Veriler kaydedildi.")
        

        # Önbellek kontrol başlıklarını ekle
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        return response, 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
