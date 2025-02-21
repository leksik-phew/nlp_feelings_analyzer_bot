import telebot
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('../model_training/sentiment_model.h5')
tokenizer = joblib.load('../model_training/tokenizer.pkl')
label_encoder = joblib.load('../model_training/label_encoder.pkl')

API_TOKEN = "7652027861:AAE-T5Mw9Jq6zwzeqpj5xEKjpg-UH19XJvA"
bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text

    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=model.input_shape[1])

    prediction = model.predict(padded_seq)
    predicted_label = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_label])[0]
    probabilities = prediction.flatten()

    response = f"Sentiment: {predicted_class}\nProbabilities:\n"
    for i, label in enumerate(label_encoder.classes_):
        response += f"{label}: {probabilities[i]:.2f}\n"

    bot.reply_to(message, response)

if __name__ == '__main__':
    bot.polling()
