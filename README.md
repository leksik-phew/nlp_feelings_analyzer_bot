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
  pip install pyarrow
  pip install fastparquet
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
  sadness: 0.01
  ```

## Обучение модели
Скрипт для обучения модели - [/model_training/train_model.py](model_training/train_model.py). Для обучения модели я использовал не файл, который приложен к заданию, а  [Emotion Dataset from Hugging Face](https://huggingface.co/datasets/dair-ai/emotion). Датасет состит из 2 колонок( Text, Label ) и содержит 6 эмоций: Sadness, Joy, Anger, Fear, Surprise, Love. Загрузить файл я не могу из-за вот такой проблемы: ```Yowza, that’s a big file. Try again with a file smaller than 25MB.```, но вы можете скачать его с помощью моего [яндекс диска](https://disk.yandex.ru/d/MerBWEFjI0-VNw): ```https://disk.yandex.ru/d/MerBWEFjI0-VNw```
