let isChatbotOpen = false;
let isFirstInteraction = true; // İlk etkileşim durumu

function toggleChatbot() {
    const widgetContainer = document.getElementById("widgetContainer");
    if (isChatbotOpen) {
        widgetContainer.style.display = "none"; // Gizle
    } else {
        widgetContainer.style.display = "block"; // Göster
        if (isFirstInteraction) {
            displayBotMessage("Merhaba, ben PassoAssist, size destek olmak için buradayım. 😊 " +
                "Eğer daha detaylı bilgi almak isterseniz, müşteri temsilcilerimiz size yardımcı olacaktır. " +
                "KVKK aydınlatma metni için sitemizi ziyaret edebilirsiniz."); // İlk etkileşim mesajı
            isFirstInteraction = false; // İlk etkileşimi bir kez gönder
        }
    }
    isChatbotOpen = !isChatbotOpen; // Durumu değiştir
}

function sendMessage(event) {
    if (event.type === 'keypress' && event.key !== 'Enter') return;

    const inputField = document.getElementById('chat-input');
    const userInput = inputField.value.trim();

    // Kullanıcı girişini kontrol et
    if (userInput === '') {
        displayBotMessage("Lütfen bir mesaj girin."); // Kullanıcıdan giriş istemek için hata mesajı
        return;
    }

    displayUserMessage(userInput);
    inputField.value = ''; // Giriş alanını temizle

    // Backend çağrısını yaparak modelden yanıt al
    fetchResponse(userInput);
}

function displayUserMessage(message) {
    const chatBody = document.getElementById('chatBody');
    const userMessageDiv = document.createElement('div');
    userMessageDiv.classList.add('user-message');
    userMessageDiv.innerHTML = `<div class="message-text">${message}</div>`;
    chatBody.appendChild(userMessageDiv);
}

function fetchResponse(userInput) {
    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: userInput }) // Sadece kullanıcı girişi gönder
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            displayBotMessage("Bir hata oluştu: " + data.error);
        } else {
            displayBotMessage(data.response_message); // Backend'den alınan yanıtı göster
        }
    })
    .catch(error => console.error('Hata:', error));
}

function displayBotMessage(message) {
    const chatBody = document.getElementById('chatBody');
    const botMessageDiv = document.createElement('div');
    botMessageDiv.classList.add('bot-message');
    botMessageDiv.innerHTML = `<img src="https://chatbot.aktifbank.com.tr/content/files/passo/icon1.gif" alt="Passo Asistan"><div class="message-text">${message}</div>`;
    chatBody.appendChild(botMessageDiv);
}
