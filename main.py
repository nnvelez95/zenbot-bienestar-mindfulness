import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# --- NLTK y Lematización ---
lemmatizer = WordNetLemmatizer()

def load_intents(file_path='intents.json'):
    """Carga las intenciones desde un archivo JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró. Asegúrate de que esté en la misma carpeta.")
        return {"intents": []}

### NUEVAS ADICIONES PARA GENERACIÓN DINÁMICA ###
def load_dynamic_phrases(file_path='dynamic_phrases.json'):
    """Carga las frases y categorías para la generación dinámica."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró. La generación dinámica no funcionará.")
        return {} # Retorna un diccionario vacío si no se encuentra

dynamic_phrases_data = load_dynamic_phrases()

def generate_dynamic_response(dynamic_data, context_tag=None):
    """
    Genera una respuesta dinámica combinando estructuras y palabras clave.
    context_tag puede usarse para influir en la selección si es necesario (futuro).
    """
    if not dynamic_data or not dynamic_data.get('estructuras_frase'):
        return "Lo siento, no puedo generar una reflexión en este momento."

    # Seleccionar una estructura de frase al azar
    structure = random.choice(dynamic_data['estructuras_frase'])
    
    # Reemplazar los marcadores de posición con palabras aleatorias de las categorías
    response = structure
    
    # Usar expresiones regulares para encontrar todos los marcadores [categoria]
    placeholders = re.findall(r'\[(\w+)\]', response)
    
    for placeholder_category in placeholders:
        if placeholder_category in dynamic_data:
            # Seleccionar una palabra al azar de la categoría correspondiente
            chosen_word = random.choice(dynamic_data[placeholder_category])
            # Reemplazar SOLO la primera ocurrencia para evitar reemplazar múltiples veces la misma categoría
            response = response.replace(f'[{placeholder_category}]', chosen_word, 1)
        else:
            # Si la categoría no existe, reemplazar con un marcador de error o vacío
            response = response.replace(f'[{placeholder_category}]', '[DESCONOCIDO]', 1)

    return response

### FIN NUEVAS ADICIONES ###


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

# --- Función para obtener respuesta (ahora usa contexto y generación dinámica) ---
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

    # --- Lógica de Generación Dinámica ---
    # Si la intención es de estrés, o si quieres una "reflexión" general
    if predicted_tag == "estado_estresado": # Puedes añadir más tags aquí, como "busqueda_calma" si la creas
        # Primero, damos una respuesta predefinida que establece el contexto
        response_options = next(item for item in intents['intents'] if item["tag"] == "estado_estresado")['responses']
        chosen_response = random.choice(response_options)
        
        # Si la respuesta elegida es una pregunta sobre el ejercicio, establecemos el contexto
        if "¿Te gustaría que probemos un ejercicio de respiración?" in chosen_response or "¿Quieres intentar uno?" in chosen_response or "¿Qué tal si hacemos una pequeña pausa y practicamos algo de mindfulness juntos?" in chosen_response:
             current_conversation_state["awaiting_specific_response"] = "respiracion_pregunta"
        else:
             current_conversation_state["awaiting_specific_response"] = None # Limpiar si no es una pregunta de seguimiento

        # Después de la respuesta predefinida, añadir una frase dinámica para un toque extra
        dynamic_phrase = generate_dynamic_response(dynamic_phrases_data, predicted_tag)
        return chosen_response + "<br><br>" + dynamic_phrase # Combina la respuesta y la frase dinámica
    
    # --- Respuestas normales basadas en la intención predicha ---
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            current_conversation_state["awaiting_specific_response"] = None
            return random.choice(intent['responses'])
            
    current_conversation_state["awaiting_specific_response"] = None
    return "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías reformularlo o preguntarme algo diferente sobre bienestar?"


# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app)

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
    app.run(debug=True, port=5000)