import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

# Asegúrate de haber descargado los datos necesarios de NLTK
# Si no los tienes, descomenta y ejecuta estas líneas una vez:
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4') # Open Multilingual Wordnet (for Spanish if needed, though 'wordnet' covers English lemmas)

lemmatizer = WordNetLemmatizer()

def load_intents(file_path='intents.json'):
    """Carga las intenciones desde un archivo JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró. Asegúrate de que esté en la misma carpeta.")
        return {"intents": []}

intents_data = load_intents()

def preprocess_text(text):
    """Limpia y lematiza el texto de entrada."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text, language='spanish')
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmas

def get_response(user_input, intents):
    """Busca la intención del usuario y devuelve una respuesta."""
    processed_input = preprocess_text(user_input)
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            processed_pattern = preprocess_text(pattern)
            
            if any(word in processed_input for word in processed_pattern if len(word) > 2):
                return random.choice(intent['responses'])
                
    return "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías reformularlo o preguntarme algo diferente sobre bienestar?"

app = Flask(__name__)
CORS(app) # Habilita CORS para todas las rutas

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para manejar las solicitudes de chat."""
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "Error: No se proporcionó ningún mensaje."}), 400
    
    bot_response = get_response(user_message, intents_data)
    return jsonify({"response": bot_response})

@app.route('/')
def index():
    """Ruta raíz para verificar que el servidor está funcionando."""
    return "El servidor del chatbot está funcionando. Usa el frontend para interactuar."

if __name__ == "__main__":
    print("Iniciando ZenBot API...")
    app.run(debug=True, port=5000)