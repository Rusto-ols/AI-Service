# ЛР №3 — исследование RAG по PDF

## Установка (Python 3.11+)

Распакуйте архив, откройте терминал в папке проекта.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements_lab3.txt
python rag_lab3.py
python -m unittest -v test_rag_lab3.py
```

Результаты всех семи заданий: `results_lab3/task1.csv` … `task7.csv` и `summary.json`. Чтобы проверить ноутбук, откройте `Лабораторная_работа_3_RAG.ipynb` в Jupyter или Colab (для Colab загрузите также `rag_lab3.py` и исходный PDF). PDF-источник находится рядом со скриптом.

## Локальная генерация Ollama

Офлайн-демонстрация выполняет реальный поиск и детерминированное извлечение подтверждения, **но не вызывает LLM**. Для проверки LLM потребуется установленный Ollama и модель:

```bash
ollama pull qwen2.5:3b
ollama serve
```

В другом терминале:

```bash
python run_ollama_demo.py
```

Для расширенной проверки двух промптов, цитирований и диалогов выполните:

```bash
python run_ollama_experiments.py
```

Реальные ответы Ollama сохраняются в `results_lab3/ollama_experiments.json` и `results_lab3/ollama_experiments.csv`. Они не подменяют уже сохранённые офлайн-результаты. Установить альтернативную модель можно через переменную окружения `OLLAMA_MODEL`.

## Методологические ограничения

- Для исходных экспериментов использованы TF-IDF символьных n-грамм + SVD (приближение плотных векторов), а не нейросетевые OllamaEmbeddings.
- «Оценка 0–2» офлайн — эвристическая проверка найденной опорной цитаты, а не человеческая оценка генерации LLM.
- В задании 6 офлайн два промпта дают одинаковый результат, так как LLM не запускалась. Скрипт `run_ollama_experiments.py` позволяет честно сравнить промпты после установки Ollama.
- В задании 7 переформулировки заданы прозрачными правилами, а отдельные сессии ведутся словарём.

## Сравнение с настоящими OllamaEmbeddings

```bash
ollama pull nomic-embed-text
python run_ollama_embeddings.py
```

Сравнение нейросетевых эмбеддингов Ollama и бинарного BoW по 10 фиксированным запросам сохраняется в `results_lab3/ollama_embeddings_task4.json`.
