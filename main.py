import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests # Para hacer peticiones HTTP a la API de Gemini

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# --- NLTK y Lematización ---
lemmatizer = WordNetLemmatizer()

# Descargar datos de NLTK al inicio de la aplicación
print("Descargando/Verificando datos de NLTK: punkt, wordnet, omw-1.4, punkt_tab...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
print("Datos de NLTK listos.")


def load_intents(file_path='intents.json'):
    """Carga las intenciones desde un archivo JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró. Asegúrate de que esté en la misma carpeta.")
        return {"intents": []}

def load_dynamic_phrases(file_path='dynamic_phrases.json'):
    """Carga las frases y categorías para la generación dinámica."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró. La generación dinámica no funcionará.")
        return {}

dynamic_phrases_data = load_dynamic_phrases()

def generate_dynamic_response(dynamic_data, context_tag=None):
    """
    Genera una respuesta dinámica combinando estructuras y palabras clave.
    context_tag puede usarse para influir en la selección si es necesario (futuro).
    """
    if not dynamic_data or not dynamic_data.get('estructuras_frase'):
        return "Lo siento, no puedo generar una reflexión en este momento."

    structure = random.choice(dynamic_data['estructuras_frase'])
    
    response = structure
    placeholders = re.findall(r'\[(\w+)\]', response)
    
    for placeholder_category in placeholders:
        if placeholder_category in dynamic_data:
            chosen_word = random.choice(dynamic_data[placeholder_category])
            response = response.replace(f'[{placeholder_category}]', chosen_word, 1)
        else:
            response = response.replace(f'[{placeholder_category}]', '[DESCONOCIDO]', 1)

    return response


def preprocess_text(text):
    """Limpia y lematiza el texto de entrada."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text, language='spanish')
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmas)

# --- Entrenamiento del Modelo de Machine Learning ---
def train_model(intents_data):
    """Entrena un modelo de clasificación de intenciones."""
    corpus = []
    labels = []

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            corpus.append(preprocess_text(pattern))
            labels.append(intent['tag'])
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('clf', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    
    pipeline.fit(corpus, labels)
    return pipeline

# --- Cargar o Entrenar el Modelo al Inicio ---
intents_data = load_intents()
model_path = 'zenbot_model.joblib'

try:
    classifier_pipeline = joblib.load(model_path)
    print("Modelo de chatbot cargado exitosamente.")
except FileNotFoundError:
    print("Modelo no encontrado. Entrenando nuevo modelo de chatbot...")
    if intents_data['intents']:
        classifier_pipeline = train_model(intents_data)
        joblib.dump(classifier_pipeline, model_path)
        print("Modelo de chatbot entrenado y guardado.")
    else:
        print("No hay datos de intenciones para entrenar el modelo.")
        classifier_pipeline = None

# --- Variable global para el contexto de la conversación ---
current_conversation_state = {"awaiting_specific_response": None}

### NUEVA FUNCIÓN: LLAMADA A LA API DE GEMINI ###
def call_gemini_api(prompt):
    """
    Realiza una llamada a la API de Gemini para generar una respuesta.
    Usa gemini-2.0-flash para texto.
    """
    # La API Key será proporcionada por el entorno de Canvas si se ejecuta allí.
    # Para Render, puedes configurar una variable de entorno en Render.
    # Por ahora, la dejamos vacía, y Render/Canvas la inyectará.
    api_key = os.environ.get("GEMINI_API_KEY", "") # Asegúrate de configurar GEMINI_API_KEY en Render

    # URL de la API de Gemini para gemini-2.0-flash
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": chat_history
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Lanza una excepción para errores HTTP (4xx o 5xx)
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print("Respuesta inesperada de la API de Gemini:", result)
            return "Lo siento, no pude generar una respuesta detallada en este momento."
    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API de Gemini: {e}")
        return "Lo siento, tengo problemas para acceder a mi conocimiento en este momento. ¿Podrías intentar de nuevo?"
    except Exception as e:
        print(f"Error inesperado al procesar la respuesta de Gemini: {e}")
        return "Hubo un problema al generar mi respuesta. Disculpa."

### FIN NUEVA FUNCIÓN ###


# --- Función para obtener respuesta (ahora usa contexto y generación dinámica/LLM) ---
def get_response(user_input, intents, classifier_pipeline):
    global current_conversation_state

    processed_input = preprocess_text(user_input)
    predicted_tag = None

    if classifier_pipeline:
        predicted_tag = classifier_pipeline.predict([processed_input])[0]
    else:
        return "Lo siento, el modelo del chatbot no está disponible. Por favor, revisa la configuración."

    # --- Lógica de Gestión de Contexto ---
    if current_conversation_state["awaiting_specific_response"] == "respiracion_pregunta":
        if predicted_tag == "afirmacion":
            current_conversation_state["awaiting_specific_response"] = None
            return random.choice(next(item for item in intents['intents'] if item["tag"] == "ejercicio_respiracion_solicitud")['responses'])
        elif predicted_tag == "negacion":
            current_conversation_state["awaiting_specific_response"] = None
            return random.choice(next(item for item in intents['intents'] if item["tag"] == "negacion")['responses'])
        current_conversation_state["awaiting_specific_response"] = None

    # --- Lógica de Generación Dinámica o LLM ---
    if predicted_tag == "estado_estresado":
        response_options = next(item for item in intents['intents'] if item["tag"] == "estado_estresado")['responses']
        chosen_response = random.choice(response_options)
        
        if "¿Te gustaría que probemos un ejercicio de respiración?" in chosen_response or "¿Quieres intentar uno?" in chosen_response or "¿Qué tal si hacemos una pequeña pausa y practicamos algo de mindfulness juntos?" in chosen_response:
             current_conversation_state["awaiting_specific_response"] = "respiracion_pregunta"
        else:
             current_conversation_state["awaiting_specific_response"] = None

        dynamic_phrase = generate_dynamic_response(dynamic_phrases_data, predicted_tag)
        return chosen_response + "<br><br>" + dynamic_phrase
    
    # --- ¡NUEVO! Lógica para Conversación Abierta con LLM ---
    elif predicted_tag == "conversacion_abierta":
        current_conversation_state["awaiting_specific_response"] = None # Limpiar contexto específico
        print(f"Llamando a Gemini API para: {user_input}") # Para depuración en logs
        llm_response = call_gemini_api(user_input)
        return llm_response # La respuesta del LLM
    
    # --- Respuestas normales basadas en la intención predicha ---
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            current_conversation_state["awaiting_specific_response"] = None
            return random.choice(intent['responses'])
            
    current_conversation_state["awaiting_specific_response"] = None
    return "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías reformularlo o preguntarme algo diferente sobre bienestar?"


# --- Configuración de Flask ---
app = Flask(__name__)

# Configuración de CORS
allowed_origins = [
    "http://localhost:8000", # Para pruebas locales con python -m http.server
    "https://nnvelez95.github.io", # Tu URL base de GitHub Pages
    "https://nnvelez95.github.io/zenbot-bienestar-mindfulness" # Tu URL completa de GitHub Pages si incluye el repo
]

CORS(app, resources={r"/*": {"origins": allowed_origins, "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

@app.route('/chat', methods=['OPTIONS'])
def handle_options():
    return '', 200

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "Error: No se proporcionó ningún mensaje."}), 400
    
    bot_response = get_response(user_message, intents_data, classifier_pipeline)
    return jsonify({"response": bot_response})

@app.route('/')
def index():
    return "El servidor del chatbot está funcionando. Usa el frontend para interactuar."

if __name__ == "__main__":
    print("Iniciando ZenBot API...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)