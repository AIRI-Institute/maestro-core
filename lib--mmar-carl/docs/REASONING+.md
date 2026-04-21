# Цепочки рассуждений с использованием CARL

## Описание методологии

**Цепочки рассуждений** -- это структурированный подход к формализации экспертного мышления, основанный на триаде **Event-Action-Result**. Методология позволяет преобразовать сложные мыслительные процессы экспертов в понятную для LLM форму, создавая основу для интеллектуальных агентных систем.

**Event** обозначает исходное событие, факт или действие, которое инициирует процесс рассуждения. **Action** отражает ответную деятельность системы или эксперта, включая уточнения, анализ и назначение процедур. **Result** показывает конечный вывод на каждом шаге, будь то полученная информация, промежуточное заключение или корректировка стратегии.

В контексте фреймворка Maestro цепочки рассуждений преобразуются в формализованные графы принятия решений, где узлы представляют состояния, действия и гипотезы, а рёбра -- причинно-следственные связи и условия переходов.

## Библиотека MMAR CARL

Для практической реализации цепочек рассуждений разработана специализированная библиотека **MMAR CARL (Collaborative Agent Reasoning Library)** -- универсальный инструмент для построения систем экспертного мышления с поддержкой:

- **RAG-подобного извлечения контекста** -- автоматическое извлечение релевантной информации из входных данных для каждого шага рассуждений
- **Параллельного выполнения на основе DAG** -- автоматическая оптимизация последовательности выполнения с учётом зависимостей между шагами
- **Мультиязычности** -- встроенная поддержка русского и английского языков с возможностью расширения
- **Универсальной архитектуры** -- применимость к любой предметной области без модификации ядра

### Ключевые компоненты

#### StepDescription -- формализация шага рассуждений

Каждый шаг цепочки описывается структурой, включающей:

```python
from mmar_carl import StepDescription

step = StepDescription(
    number=1,
    title="Оценка исходных данных",
    aim="Провести анализ качества и полноты входных данных",
    reasoning_questions="Какие аномалии и паттерны присутствуют в данных?",
    step_context_queries=[
        "показатели качества данных",
        "пропущенные значения",
        "согласованность данных"
    ],
    stage_action="Оценить надёжность данных и выявить потенциальные проблемы",
    example_reasoning="Высокое качество данных обеспечивает более надёжный анализ",
    dependencies=[]  # Список номеров шагов-предшественников
)
```

**Структура шага:**

- `number` -- уникальный номер шага в цепочке
- `title` -- краткое название этапа рассуждений
- `aim` -- цель выполнения данного шага
- `reasoning_questions` -- ключевые вопросы для анализа
- `step_context_queries` -- запросы для извлечения релевантного контекста из входных данных
- `stage_action` -- конкретные действия, выполняемые на этапе
- `example_reasoning` -- пример экспертного рассуждения для данного шага
- `dependencies` -- список номеров шагов, которые должны быть выполнены до текущего

#### ReasoningChain -- оркестрация выполнения

Класс для управления последовательностью рассуждений с оптимизацией параллельного выполнения:

```python
from mmar_carl import ReasoningChain

chain = ReasoningChain(
    steps=[step1, step2, step3],
    max_workers=2,  # Максимальное количество параллельных потоков
    enable_progress=True  # Отображение прогресса выполнения
)
```

#### ReasoningContext -- контекст выполнения

Управляет состоянием, историей выполнения и интеграцией с LLM:

```python
from mmar_carl import ReasoningContext, Language

context = ReasoningContext(
    outer_context=input_data,  # Исходные данные для анализа
    model="gigachat-2-max",
    language=Language.RUSSIAN,  # Язык рассуждений
    retry_max=3                # Максимальное количество повторов при ошибках
)
```

## Процесс разработки цепочек рассуждений

### 1. Извлечение и структурирование экспертного знания

Формализация экспертного мышления в структурированный формат:

```python
MEDICAL_DIAGNOSIS_CHAIN = [
    StepDescription(
        number=1,
        title="Сбор анамнеза",
        aim="Собрать полную информацию о состоянии пациента",
        reasoning_questions="Какие симптомы и их длительность указывают на диагноз?",
        step_context_queries=[
            "жалобы пациента",
            "история заболевания",
            "сопутствующие патологии"
        ],
        stage_action="Систематизировать жалобы и анамнез",
        example_reasoning="Длительность и характер симптомов помогают дифференцировать острые и хронические состояния"
    ),
    StepDescription(
        number=2,
        title="Анализ объективных данных",
        aim="Оценить результаты физикального осмотра и лабораторных исследований",
        reasoning_questions="Какие объективные признаки подтверждают или опровергают предварительные гипотезы?",
        dependencies=[1],  # Зависит от сбора анамнеза
        step_context_queries=[
            "результаты осмотра",
            "лабораторные показатели",
            "инструментальные данные"
        ],
        stage_action="Сопоставить субъективные и объективные данные",
        example_reasoning="Отклонения в лабораторных показателях коррелируют с клинической картиной"
    )
]
```

### 2. Построение графа зависимостей

CARL автоматически анализирует зависимости и создаёт оптимальный граф выполнения:

```python
# Параллельное выполнение независимых шагов
step1 = StepDescription(number=1, title="Анализ выручки", dependencies=[])
step2 = StepDescription(number=2, title="Анализ затрат", dependencies=[])

# Последовательное выполнение зависимых шагов
step3 = StepDescription(
    number=3,
    title="Анализ рентабельности",
    dependencies=[1, 2]  # Ждёт завершения шагов 1 и 2
)

# Шаги 1 и 2 выполняются параллельно, затем выполняется шаг 3
```

### 3. RAG-подобное извлечение контекста

Для каждого шага автоматически извлекается релевантная информация из входных данных:

```python
step = StepDescription(
    number=1,
    title="Финансовый анализ",
    aim="Проанализировать финансовые показатели",
    reasoning_questions="Каковы ключевые тренды в финансовой отчётности?",
    step_context_queries=[
        "рост выручки",
        "маржинальность",
        "операционная эффективность"
    ],
    stage_action="Рассчитать финансовые коэффициенты и тренды",
    example_reasoning="Финансовый анализ выявляет устойчивость бизнеса"
)

# CARL автоматически извлекает релевантные фрагменты из outer_context
# и включает их в промпт для LLM
```

## Конфигурация поиска контекста

CARL поддерживает два режима поиска релевантной информации:

### Подстроковый поиск (по умолчанию)

Быстрый текстовый поиск без дополнительных зависимостей:

```python
from mmar_carl import ContextSearchConfig, ReasoningChain

search_config = ContextSearchConfig(
    strategy="substring",
    substring_config={
        "case_sensitive": False,  # Регистронезависимый поиск
        "min_word_length": 3,     # Минимальная длина слова
        "max_matches_per_query": 5  # Максимум результатов на запрос
    }
)

chain = ReasoningChain(
    steps=steps,
    search_config=search_config
)
```

### Векторный поиск с FAISS

Семантический поиск на основе эмбеддингов и векторного сходства:

```python
search_config = ContextSearchConfig(
    strategy="vector",
    embedding_model="all-MiniLM-L6-v2",
    vector_config={
        "index_type": "flat",  # или "ivf" для больших массивов данных
        "similarity_threshold": 0.7,  # Порог семантического сходства
        "max_results": 5
    }
)

chain = ReasoningChain(
    steps=steps,
    search_config=search_config
)
```

### Индивидуальная конфигурация для запросов

Гибкое управление стратегией поиска для отдельных запросов:

```python
from mmar_carl import ContextQuery

step = StepDescription(
    number=1,
    title="Комплексный анализ",
    aim="Анализ с различными стратегиями поиска",
    reasoning_questions="Какие инсайты можно извлечь?",
    stage_action="Извлечь комплексную картину",
    example_reasoning="Комбинированный поиск обеспечивает полноту анализа",
    step_context_queries=[
        "EBITDA",  # Простая строка (использует настройки цепочки)
        ContextQuery(
            query="тренды выручки",
            search_strategy="vector",
            search_config={
                "similarity_threshold": 0.8,
                "max_results": 3
            }
        ),
        ContextQuery(
            query="ЧИСТАЯ_ПРИБЫЛЬ",
            search_strategy="substring",
            search_config={
                "case_sensitive": True,
                "min_word_length": 4
            }
        )
    ]
)
```

## Многоязычная поддержка

Встроенная поддержка русского и английского языков с автоматическим выбором шаблонов промптов:

```python
# Рассуждения на русском языке
context_ru = ReasoningContext(
    outer_context=data,
    model="gigachat-2-max",
    language=Language.RUSSIAN
)

# Рассуждения на английском языке
context_en = ReasoningContext(
    outer_context=data,
    model="gigachat-2-max",
    language=Language.ENGLISH
)
```

## Пример полного цикла работы

### Определение цепочки рассуждений

```python
import asyncio
from mmar_carl import (
    ReasoningChain, StepDescription, ReasoningContext,
    Language, ContextSearchConfig
)

# Формализация экспертных знаний
FINANCIAL_ANALYSIS = [
    StepDescription(
        number=1,
        title="Оценка исходных данных",
        aim="Проверить качество и полноту финансовых данных",
        reasoning_questions="Присутствуют ли аномалии или пропуски в данных?",
        step_context_queries=[
            "индикаторы качества данных",
            "пропущенные значения",
            "согласованность данных"
        ],
        stage_action="Оценить надёжность данных",
        example_reasoning="Качественные данные обеспечивают точность анализа"
    ),
    StepDescription(
        number=2,
        title="Анализ трендов",
        aim="Выявить значимые паттерны и тренды",
        reasoning_questions="Какие тренды и корреляции проявляются?",
        dependencies=[1],
        step_context_queries=[
            "тренды роста",
            "показатели эффективности",
            "паттерны корреляций"
        ],
        stage_action="Проанализировать временные паттерны",
        example_reasoning="Распознавание паттернов выявляет драйверы бизнеса"
    ),
    StepDescription(
        number=3,
        title="Формирование рекомендаций",
        aim="Сформулировать стратегические рекомендации",
        reasoning_questions="Какие действия следует предпринять?",
        dependencies=[1, 2],
        step_context_queries=[
            "возможности роста",
            "области риска",
            "стратегические приоритеты"
        ],
        stage_action="Синтезировать выводы и рекомендации",
        example_reasoning="Рекомендации основываются на комплексном анализе данных и трендов"
    )
]
```

### Конфигурация и выполнение

```python
# Настройка векторного поиска для семантического анализа
search_config = ContextSearchConfig(
    strategy="vector",
    vector_config={
        "similarity_threshold": 0.75,
        "max_results": 5
    }
)

# Создание цепочки с параллельным выполнением
chain = ReasoningChain(
    steps=FINANCIAL_ANALYSIS,
    search_config=search_config,
    max_workers=2,
    enable_progress=True
)

# Подготовка входных данных
financial_data = """
Период,Выручка,Прибыль,Сотрудники
2023-Q1,1000000,200000,50
2023-Q2,1200000,300000,55
2023-Q3,1100000,250000,52
2023-Q4,1400000,400000,60
"""

# Системный промпт для финансовой экспертизы
financial_system_prompt = """
Вы старший финансовый аналитик с 15-летним опытом в корпоративных финансах.

Ваш анализ должен:
- Основываться на данных и доказательствах
- Включать конкретные проценты и тренды
- Предоставлять практические выводы и рекомендации
- Учитывать отраслевые стандарты и лучшие практики
- Сохранять профессиональную объективность
"""

# Создание контекста выполнения
context = ReasoningContext(
    outer_context=financial_data,
    model="gigachat-2-max",
    language=Language.RUSSIAN,
    retry_max=3,
    system_prompt=financial_system_prompt.strip()
)

# Выполнение цепочки рассуждений
result = chain.execute(context)

# Получение результатов
final_output = result.get_final_output()
print(final_output)

# Доступ к результатам отдельных шагов
for step_num, step_result in result.step_results.items():
    print(f"Шаг {step_num}: {step_result}")
```

## Применение в различных доменах

### Медицинская диагностика

```python
CLINICAL_REASONING = [
    StepDescription(
        number=1,
        title="Сбор жалоб и анамнеза",
        aim="Систематизировать клиническую картину",
        reasoning_questions="Какие симптомы указывают на возможные диагнозы?",
        step_context_queries=[
            "основные жалобы",
            "длительность симптомов",
            "факторы риска"
        ],
        stage_action="Формирование предварительных гипотез",
        example_reasoning="Характер и динамика симптомов определяют направление диагностического поиска"
    ),
    StepDescription(
        number=2,
        title="Назначение обследований",
        aim="Подтвердить или опровергнуть диагностические гипотезы",
        reasoning_questions="Какие исследования необходимы для дифференциальной диагностики?",
        dependencies=[1],
        step_context_queries=[
            "клинические рекомендации",
            "протоколы обследования",
            "дифференциальная диагностика"
        ],
        stage_action="Планирование диагностических процедур",
        example_reasoning="Набор исследований соответствует дифференциально-диагностическому ряду"
    )
]
```

### Юридический анализ

```python
LEGAL_ANALYSIS = [
    StepDescription(
        number=1,
        title="Анализ фактических обстоятельств",
        aim="Установить юридически значимые факты",
        reasoning_questions="Какие факты имеют правовое значение?",
        step_context_queries=[
            "обстоятельства дела",
            "доказательства",
            "стороны спора"
        ],
        stage_action="Квалификация фактических обстоятельств",
        example_reasoning="Юридическая квалификация основывается на доказанных фактах"
    ),
    StepDescription(
        number=2,
        title="Применение норм права",
        aim="Определить применимые правовые нормы",
        reasoning_questions="Какие нормы регулируют данные правоотношения?",
        dependencies=[1],
        step_context_queries=[
            "применимое законодательство",
            "судебная практика",
            "доктринальные позиции"
        ],
        stage_action="Правовая квалификация спора",
        example_reasoning="Применение норм основывается на установленных фактах и правовых позициях"
    )
]
```

## Оценка качества: MetricBase

CARL поддерживает подключение произвольных метрик к шагам и к цепочке в целом.
Метрика реализует абстрактный класс `MetricBase` — принимает текстовый вывод и возвращает числовое значение.
Внутри может быть что угодно: подсчёт слов, парсинг файла, косинусное сходство, LLM-as-a-judge.

### Абстрактный класс

```python
from mmar_carl import MetricBase

class WordCountMetric(MetricBase):
    @property
    def name(self) -> str:
        return "word_count"          # ключ в словаре результатов

    async def compute_async(self, text: str) -> float:
        return float(len(text.split()))
```

**Контракт:**

- `name` (property) — уникальное имя; используется как ключ в `StepExecutionResult.metrics` / `ReasoningResult.metrics`
- `compute_async(text)` — асинхронный метод, возвращает `float`; вызывается после успешного выполнения шага / цепочки
- `compute(text)` — синхронная обёртка через `asyncio.run()`

### Метрика на шаге

```python
from mmar_carl import LLMStepDescription

step = LLMStepDescription(
    number=1,
    title="Анализ данных",
    aim="Провести анализ финансовых показателей",
    metrics=[WordCountMetric()],
)
```

После выполнения шага оценки доступны в `StepExecutionResult.metrics`:

```python
result = chain.execute(context)
print(result.step_results[0].metrics)
# {'word_count': 47.0}
```

### Метрика на цепочке

Цепочковые метрики вычисляются на финальном выводе (`get_final_output()`):

```python
chain = ReasoningChain(
    steps=steps,
    metrics=[WordCountMetric(), KeywordCoverageMetric(["риск", "рост"])],
)

result = chain.execute(context)
print(result.metrics)
# {'word_count': 82.0, 'keyword_coverage': 0.5}
```

### Примеры готовых метрик

```python
class SentenceLengthMetric(MetricBase):
    """Среднее число слов в предложении."""

    @property
    def name(self) -> str:
        return "avg_sentence_length"

    async def compute_async(self, text: str) -> float:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)


class KeywordCoverageMetric(MetricBase):
    """Доля найденных ключевых слов (0.0 – 1.0)."""

    def __init__(self, keywords: list[str]):
        self._keywords = [kw.lower() for kw in keywords]

    @property
    def name(self) -> str:
        return "keyword_coverage"

    async def compute_async(self, text: str) -> float:
        if not self._keywords:
            return 1.0
        lowered = text.lower()
        found = sum(1 for kw in self._keywords if kw in lowered)
        return found / len(self._keywords)


class LLMJudgeMetric(MetricBase):
    """LLM-as-a-judge: запрашивает LLM оценить текст и возвращает число."""

    def __init__(self, llm_client, scale: int = 10):
        self._client = llm_client
        self._scale = scale

    @property
    def name(self) -> str:
        return "llm_judge_score"

    async def compute_async(self, text: str) -> float:
        prompt = (
            f"Rate the quality of the following text on a scale from 0 to {self._scale}. "
            f"Return only a single number.\n\nText:\n{text}"
        )
        response = await self._client.get_response(prompt)
        import re
        match = re.search(r"\d+(?:\.\d+)?", response)
        return float(match.group()) if match else 0.0
```

### Поведение при ошибках

- Метрика вычисляется только для **успешно** завершённых шагов; шаги с ошибками пропускаются
- Исключение внутри `compute_async` **не прерывает** выполнение цепочки — ошибка логируется, метрика пропускается
- Множество метрик на одном шаге независимы: сбой одной не влияет на остальные

### Сериализация

Числовые оценки включаются в `to_dict()`:

```python
result.to_dict()["metrics"]                       # оценки цепочки
result.to_dict()["step_results"][0]["metrics"]    # оценки шага
```

Сами объекты метрик из JSON сериализации **исключены** (`exclude=True`) — они содержат исполняемый код.

### Метрики и рефлексия

При вызове `chain.reflect()` результаты метрик автоматически включаются в промпт рефлексии
(через `ReflectionOptions.include_metric_scores=True`, включено по умолчанию).
LLM видит конкретные числовые сигналы качества и может ссылаться на них в рекомендациях.

Дополнительно пользователь может передать произвольный контекст через `ReflectionOptions.extra_feedback`:

```python
from mmar_carl import ReflectionOptions

result = chain.execute(context)

# Метрики автоматически попадают в промпт рефлексии
reflection = chain.reflect(
    task_description="Анализ финансовых данных",
    options=ReflectionOptions(
        include_metric_scores=True,          # True по умолчанию
        extra_feedback={                     # опциональный пользовательский контекст
            "domain": "финансовый анализ",
            "audience": "совет директоров",
            "priority": "лаконичность",
        },
    ),
)
```

`extra_feedback` принимает как словарь (с метками), так и обычную строку:

```python
options = ReflectionOptions(
    extra_feedback="Фокус на шаге 2: улучшить охват ключевых слов.",
)
```

| Параметр                | Тип                     | Описание                                      |
| ----------------------- | ----------------------- | --------------------------------------------- |
| `include_metric_scores` | `bool` (default `True`) | Включить оценки MetricBase в промпт рефлексии |
| `extra_feedback`        | `dict \| str \| None`   | Дополнительный пользовательский контекст      |

Подробный пример: `examples/reflection_metrics_example.py` (`make example-reflection-metrics`).

## Преимущества использования CARL

- **🎯 Формализация экспертного мышления** -- структурированное представление сложных рассуждений
- **⚡ Автоматическая оптимизация** -- параллельное выполнение независимых шагов
- **🔍 Интеллектуальное извлечение контекста** -- RAG-подобный поиск релевантной информации
- **🌍 Многоязычность** -- встроенная поддержка русского и английского
- **🏗️ Универсальность** -- применимость к любой предметной области
- **🔧 Гибкость конфигурации** -- тонкая настройка стратегий поиска
- **📊 Оценка качества** -- подключение метрик к шагам и цепочкам через `MetricBase`
- **⚙️ Готовность к продакшену** -- обработка ошибок, повторные попытки, мониторинг
- **🔗 Простая интеграция** -- прямая работа с mmar-llm без излишних абстракций
