# Maestro

Сервис распознавания текста из документов и изображений для Цифрового Помощника

## Ответственный разработчик

@glazkov

## Общая информация

### Фичи

- чтение из pdf с текстовым слоем
- чтение текста с изображений
- чтение текста из pdf со сканами
- извлечение таблиц в формате Markdown
- извлечение описаний изображений с помощью VLM (планируется)

### Линтеры

```shell
pip install black flake8-pyproject mypy
black .
flake8
mypy .
```

или через pre-commit

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files # проверка вручную
```

## Развертка

### Branch

- dev: `sudo docker-compose -f docker-compose-dev.yaml up --build`
- prod: `sudo docker-compose -f docker-compose.yaml up --build`

## Зависимости

```sh
apt install tesseract-ocr-rus tesseract-ocr-eng -y
python3.13 -m pip install -r requirements.txt
```

## Конфигурация приложения

```sh
src/config.py
```
