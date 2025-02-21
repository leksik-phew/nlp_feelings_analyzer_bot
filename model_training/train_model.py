import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import joblib

# Загружаю данные из файла train.txt с разделителем ';' и указанием названий столбцов
data = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'label'])

# Кодирую метки классов в числовые значения
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Разделяю данные на обучающую и тестовую выборки (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Создаю токенизатор для преобразования текстов в последовательности чисел, ограничиваю на 1000 наиболее частых слов и обучаю токенизатор
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)

# Преобразую текста в последовательности чисел
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Определяю максимальную длину последовательности для дополнения и дополняю последовательности до одинаковой длины
max_length = max(len(x) for x in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Создаю модель
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=max_length), # Встраивание слов в пространство размерности 64
    tf.keras.layers.GlobalAveragePooling1D(), # Глобальное усреднение по временной оси
    tf.keras.layers.Dense(64, activation='relu'), # Полносвязный слой с 64 нейронами и активацией ReLU
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax') # Выходной слой с количеством нейронов равным количеству классов и активацией softmax
])

# Компилирую модели с указанием функции потерь, оптимизатора и метрик
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаю модель на обучающих данных с валидацией на тестовых данных
model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test))

# Сохраненяю обученную модель, токенизатор и кодировщик меток в файлы
model.save('sentiment_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Оцениваю модель на тестовых данных и вывожу точность
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Accuracy: {accuracy:.2f}')
