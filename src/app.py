# app.py
# Flask uygulamasını başlatan ana dosya.
# API endpoint'lerini burada tanımlayabiliriz.

from flask import Flask, jsonify
from src.api.routes import api_blueprint

app = Flask(__name__)

# API blueprint'ini ekleyelim
app.register_blueprint(api_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
