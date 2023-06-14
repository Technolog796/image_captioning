# Image Captioning

Создание модели для image captioning и получения качественных эмбеддингов для решения других задач в zero shot в рамках весеннего проекта, ПАДиИИ ВШЭ СПб, весна 2023

## Воспроизведение результатов

Для обучения модели достаточно запустить файл <code>train.py</code>, указав нужные параметры в <code>config.json</code>

## Структура проекта
<code>LLM_train</code> - ноутбуки для перевода англоязычного датасета и дальнейшего обучения на нём языковой модели-декодера.

<code>datasets</code> - всё, что использовалалось для подготовки данных для модели:
    1. <code>clipscore_quality_test</code> - ручной тест качества метрики clipscore и переводчика.
    2. <code>coco_translation</code> - перевод caption'ов COCO-2014.
    3. <code>flan_translation</code> - перевод CoT части FLAN'a с английского на русский.
    4. <code>load_data</code> - загрузка изображений из url'ов датасета Wiki и скачивание COCO-2014
    5. <code>wikitext_normalization</code> - нормализация и предобработка caption'ов Wiki датасета и подсчет CLIP score его и COCO.

<code>experiments</code> - все предыдущие эксперименты с моделью и её производными

<code>src</code> - актуальная модель

<code>telegram bot</code> - весь код, связанный с телеграм-ботом. Он использует следующий скрипт в качестве основы для работы с моделью: <code>experiments/inference_clip_gpt2_coco</code>

## Демо
Вы можете попробовать модель по следующим ссылкам:
1. [Telegram](https://t.me/multimodal_image_bot)
2. [HF Spaces](https://huggingface.co/spaces/Anonumous/RuImageCaptioning)
