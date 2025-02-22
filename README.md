# nlp_feelings_analyzer_bot

Этот проект представляет собой Telegram-бота, который использует модель машинного обучения для анализа настроений текстовых сообщений. Бот принимает текстовые сообщения на английском языке от пользователей и возвращает предсказание о настроении, а также вероятности для каждой эмоции.

## Функциональность

- Принимает текстовые сообщения от пользователей.
- Использует обученную модель для определения настроения текста.
- Возвращает предсказанное настроение и вероятности для каждой эмоции.

## Использование
#### Username бота - @nlp_feelings_analyzer_bot

Чтобы запустить телеграмм-бота достаточно:
1. Cкачать репозиторий
2. Установить зависимости:
 ```bash
  pip install telebot
  pip install joblib
  pip install tensorflow
  pip install pandas
  pip install scikit-learn
  ```
3. Запустить файл [/telegram-bot/bot.py](telegram_bot/bot.py)
- Модель уже обучена, никаких дополнительный действий не требуется.
- Если у вас возникла ошибка при установке *tenserflow*, воспользуйтесь ```https://pythonguides.com/install-tensorflow/```

### Пример использования
- Сообщение пользователя: I'm proud of my achievements
- Ответ бота:
  ```
  Sentiment: joy
  Probabilities:
  anger: 0.02
  fear: 0.00
  joy: 0.96
  neutral: 0.00
  sadness: 0.01
  ```

## Обучение модели
Скрипт для обучения модели - [/model_training/train_model.py](model_training/train_model.py). Для обучения модели я использовал файл [/model_training/train.txt](/model_training/train.txt), который содержит фразы форматом: {phrase};{feeling}. Скрипт нацелен на обучение с помощью таких файлов. 
