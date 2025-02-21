# nlp_feelings_analyzer_bot
Телеграмм-бот, анализирующий чувства в сообщении на английском языке

## Использование

Чтобы запустить телеграмм-бота достаточно скачать все файлы и запустить файл [/telegram-bot/bot.py](telegram_bot/bot.py). Модель уже обучена, никаких дополнительный действий не требуется.
#### Username бота - @nlp_feelings_analyzer_bot

### Важно
Перед запуском необходимо установить зависимости:
```bash
pip install telebot
pip install joblib
pip install tensorflow
pip install pandas
pip install scikit-learn
```

## Обучение модели
Скрипт для обучения модели - [/model_training/train_model.py](model_training/train_model.py). Для обучения модели я использовал файл [/model_training/train.txt](/model_training/train.txt), который содержит фразы форматом: {phrase};{feeling}. Скрипт нацелен на обучение с помощью таких файлов. 
