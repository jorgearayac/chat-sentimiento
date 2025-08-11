import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar modelo y tokenizer
model = tf.keras.models.load_model("model/sentiment_model.keras")

with open("model/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Función de preprocesamiento
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # quitar puntuación
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)  # mismo maxlen usado en entrenamiento
    return padded

# Función de predicción
def predict_sentiment(text):
    processed = preprocess_text(text)
    prediction = model.predict(processed)[0][0]  # valor entre 0 y 1
    if prediction >= 0.5:
        sentiment = "Positivo"
    else:
        sentiment = "Negativo"
    return sentiment, prediction

# Probar con ejemplos
if __name__ == "__main__":
    while True:
        user_input = input("Escribe una frase (o 'salir' para terminar): ")
        if user_input.lower() == "salir":
            break
        sentiment, score = predict_sentiment(user_input)
        print(f"Sentimiento: {sentiment} (Confianza: {score:.2f})\n")
