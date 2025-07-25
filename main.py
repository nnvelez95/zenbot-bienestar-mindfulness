import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

# Importaciones para Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib # Para guardar y cargar el modelo

# --- NLTK y Lematización ---
# Asegúrate de haber descargado los datos necesarios de NLTK
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4') # Open Multilingual Wordnet (for Spanish if needed)

lemmatizer = WordNetLemmatizer()

def load_intents(file_path='intents.json'):
    """Carga las intenciones desde un archivo JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró. Asegúrate de que esté en la misma carpeta.")
        return {"intents": []}

# --- Funciones de Preprocesamiento de Texto ---
def preprocess_text(text):
    """Limpia y lematiza el texto de entrada."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Eliminar puntuación
    tokens = nltk.word_tokenize(text, language='spanish') # Tokenizar
    lemmas = [lemmatizer.lemmatize(word) for word in tokens] # Lematizar
    return " ".join(lemmas) # Unir lemas para la vectorización

# --- Entrenamiento del Modelo de Machine Learning ---
def train_model(intents_data):
    """Entrena un modelo de clasificación de intenciones."""
    corpus = [] # Almacenará los patrones preprocesados
    labels = [] # Almacenará las etiquetas (tags) de las intenciones

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            corpus.append(preprocess_text(pattern))
            labels.append(intent['tag'])
    
    # Crea un pipeline: primero vectoriza el texto, luego aplica un clasificador
    # LogisticRegression es un buen clasificador para empezar, es rápido y efectivo
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))), # ngram_range(1,2) considera palabras individuales y pares de palabras
        ('clf', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    
    pipeline.fit(corpus, labels)
    return pipeline

# --- Cargar o Entrenar el Modelo al Inicio ---
# Esto asegura que el modelo esté disponible cuando el servidor Flask se inicie.
intents_data = load_intents()
model_path = 'zenbot_model.joblib' # Nombre del archivo para guardar el modelo

try:
    # Intenta cargar el modelo si ya existe
    classifier_pipeline = joblib.load(model_path)
    print("Modelo de chatbot cargado exitosamente.")
except FileNotFoundError:
    # Si no existe, entrena uno nuevo
    print("Modelo no encontrado. Entrenando nuevo modelo de chatbot...")
    if intents_data['intents']: # Asegúrate de que haya datos para entrenar
        classifier_pipeline = train_model(intents_data)
        joblib.dump(classifier_pipeline, model_path) # Guarda el modelo entrenado
        print("Modelo de chatbot entrenado y guardado.")
    else:
        print("No hay datos de intenciones para entrenar el modelo.")
        classifier_pipeline = None # El chatbot no podrá clasificar intenciones

# --- Función para obtener respuesta (ahora usa el modelo) ---
def get_response(user_input, intents, classifier_pipeline):
    """Predice la intención del usuario usando el modelo y devuelve una respuesta."""
    if not classifier_pipeline:
        return "Lo siento, el modelo del chatbot no está disponible. Por favor, revisa la configuración."

    processed_input = preprocess_text(user_input)
    
    # Predecir la intención
    predicted_tag = classifier_pipeline.predict([processed_input])[0]
    
    # Podemos añadir un umbral de confianza si queremos hacer el bot más "seguro"
    # Por ahora, simplemente usamos la intención predicha.
    
    # Buscar la respuesta para la intención predicha
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
            
    # Esto no debería ocurrir si el modelo está bien entrenado y el tag existe
    return "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías reformularlo o preguntarme algo diferente sobre bienestar?"


# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app) # Habilita CORS para todas las rutas

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para manejar las solicitudes de chat."""
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