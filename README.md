# image_captioning
Исследование мультимодальности в image2text задачах, ПАДиИИ ВШЭ СПб, весна 2023
# Структура
В папке <code>research</code> приведен весь код, связанный с самой моделью.
1. <code>baseline_qa_gpt-exp</code> - итоговый(на данный момент) вариант модели с использованием sber-GPT3-medium в качестве языковой модели, CLIP ViT-B-16-plus-240 и ruCOCO в качестве датасета
2. <code>baseline_qa_gpt</code> - итоговый(на данный момент) вариант модели с использованием sber-GPT3-medium в качестве языковой модели, CLIP ViT-L/14@336px и ruCOCO в качестве датасета
3. <code>baseline_qa_gpt_neo</code> - эксперимент с GPTNeo и английским COCO
4. <code>baseline_qa_mt0</code> - эксперимент с MT0 и английским COCO
5. В папке <code>research/data</code> ноутбуки с загрузкой и предобработкой данных:
    1. <code>load_data</code> - загрузка изображений из url'ов датасета Wiki и скачивание COCO-2014
    2. <code>wikitext_normalization</code> - нормализация и предобработка caption'ов Wiki датасета и подсчет CLIP score его и COCO.

В папке <code>telegram bot</code> приведен весь код, связанный с телеграм-ботом. Он использует скрипт <code>inference_clip_gpt2_coco</code> в качестве основы для работы с моделью.

# Демо
Вы можете попробовать модель по следующим ссылкам:
1. [Telegram](https://t.me/multimodal_image_bot)
2. [HF Spaces](https://huggingface.co/spaces/Anonumous/RuImageCaptioning)
