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
# Guarda el estado de la conversación para la sesión actual.
# 'awaiting_specific_response': Indica si el bot espera una respuesta particular a una pregunta anterior.
# Por ejemplo, 'respiracion_pregunta' si el bot acaba de preguntar si el usuario quiere un ejercicio de respiración.
current_conversation_state = {"awaiting_specific_response": None}

# --- Función para obtener respuesta (ahora usa contexto) ---
def get_response(user_input, intents, classifier_pipeline):
    global current_conversation_state # Accede a la variable global

    processed_input = preprocess_text(user_input)
    predicted_tag = None

    if classifier_pipeline:
        # Primero, intentar predecir la intención general del usuario
        predicted_tag = classifier_pipeline.predict([processed_input])[0]
    else:
        # Esto solo debería pasar si el modelo no se cargó/entrenó correctamente
        return "Lo siento, el modelo del chatbot no está disponible. Por favor, revisa la configuración."

    # --- Lógica de Gestión de Contexto ---

    # 1. Evaluar si se está respondiendo a una pregunta específica
    if current_conversation_state["awaiting_specific_response"] == "respiracion_pregunta":
        if predicted_tag == "afirmacion": # Si el usuario dijo "sí", "claro", etc.
            current_conversation_state["awaiting_specific_response"] = None # Limpiar el estado
            # Retorna la respuesta del ejercicio de respiración
            return random.choice(next(item for item in intents['intents'] if item["tag"] == "ejercicio_respiracion_solicitud")['responses'])
        elif predicted_tag == "negacion": # Si el usuario dijo "no", "nada", etc.
            current_conversation_state["awaiting_specific_response"] = None # Limpiar el estado
            return random.choice(next(item for item in intents['intents'] if item["tag"] == "negacion")['responses'])
        # Si no es una afirmación ni negación, podría ser una nueva pregunta o un cambio de tema.
        # En este caso, dejaremos que el flujo normal continúe, pero podríamos añadir más lógica.
        current_conversation_state["awaiting_specific_response"] = None # Resetear el estado si no fue una respuesta directa

    # 2. Establecer contexto si la intención predicha lo requiere
    # Si el bot acaba de preguntar sobre el estrés, establecerá el contexto para la próxima interacción
    if predicted_tag == "estado_estresado":
        # Buscamos la intención para obtener la respuesta. Esta respuesta debería terminar con una pregunta.
        response_options = next(item for item in intents['intents'] if item["tag"] == "estado_estresado")['responses']
        chosen_response = random.choice(response_options)
        
        # Si la respuesta elegida es una pregunta sobre el ejercicio, establecemos el contexto
        if "¿Te gustaría que probemos un ejercicio de respiración?" in chosen_response or "¿Quieres intentar uno?" in chosen_response or "¿Qué tal si hacemos una pequeña pausa y practicamos algo de mindfulness juntos?" in chosen_response:
             current_conversation_state["awaiting_specific_response"] = "respiracion_pregunta"
        else:
             current_conversation_state["awaiting_specific_response"] = None # Limpiar si no es una pregunta de seguimiento

        return chosen_response
    
    # 3. Respuestas normales basadas en la intención predicha (si no hay contexto activo o no se aplicó)
    # Para cualquier otra intención, simplemente buscamos la respuesta.
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            current_conversation_state["awaiting_specific_response"] = None # Limpiar el contexto si la nueva intención no requiere seguimiento
            return random.choice(intent['responses'])
            
    # Último recurso si no se encuentra la intención (raro con el clasificador entrenado)
    current_conversation_state["awaiting_specific_response"] = None
    return "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías reformularlo o preguntarme algo diferente sobre bienestar?"


# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app) # Habilita CORS para todas las rutas

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