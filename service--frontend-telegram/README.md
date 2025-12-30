# Telegram Front
Код для запуска телеграм бота Цифрового Помощника
Обращается к сервису geteway

## Ответственный разработчик
@evjava

## Общая информация
Сервис содержит логику, связанную с
- взаимодействием с Telegram
- авторизацией

Остальное пробрасывается на бэкенд, Maestro.
## Тесты
### Линтеры
```shell
pip install black flake8-pyproject pre-commit
black .
flake8
```

или через pre-commit

```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files # проверка вручную
```

## Развертка
### Branch
- dev: `sudo docker-compose -f docker-compose-dev.yaml up --build`
- prod: `sudo docker-compose -f docker-compose.yaml up --build`

## Зависимости
`python -m pip install -r requirements.txt`
