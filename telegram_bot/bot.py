import telebot
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка модели
model = tf.keras.models.load_model('../model_training/sentiment_model.h5')
tokenizer = joblib.load('../model_training/tokenizer.pkl')
label_encoder = joblib.load('../model_training/label_encoder.pkl')

# Инициализация телеграмм-бота
API_TOKEN = "7652027861:AAE-T5Mw9Jq6zwzeqpj5xEKjpg-UH19XJvA" 
bot = telebot.TeleBot(API_TOKEN)

# Функция для обработки сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text

    # Обрабатываю сообщение
    if text == '/start':
        bot.reply_to(message, 'Hi, send me a text and Ill tell you what emotion is inherent in it.')
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(seq, maxlen=model.input_shape[1])
    
        # Получаем ответ модели и высчитываем вероятности эмоций
        prediction = model.predict(padded_seq)
        predicted_label = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_label])[0]
        probabilities = prediction.flatten()

        # Вычисляю вероятность эмоции в процентах
        percent_probabilities = []
        for i in probabilities:
            percent = round(float(i) * 100, 2)
            percent_probabilities.append(percent)

        # Эмоции в текстовом виде (anger, fear, joy, love, sadness, surprise)
        emojies = ['\U0001F621', '\U0001F631', '\U0001F603', '\U0001F60D', '\U0001F614', '\U0001F62E']
    
        # Формируем ответ модели
        response = f"Sentiment: {predicted_class}{emojies[predicted_label]}\n"
        response += "Probabilities:\n"
        for i, label in enumerate(label_encoder.classes_):
            response += f"{emojies[i]}: {percent_probabilities[i]}%\n"
    
        # Отвечаем на сообщение пользователя
        bot.reply_to(message, response)

if __name__ == '__main__':
    bot.polling()
