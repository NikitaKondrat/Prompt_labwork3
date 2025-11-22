import sys
import types

# Создаём "заглушку" для langchain.debug
if "langchain" not in sys.modules:
    fake_langchain = types.ModuleType("langchain")
    fake_langchain.debug = False  # или True — не важно, главное чтобы атрибут существовал
    sys.modules["langchain"] = fake_langchain

import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from typing import List
import warnings
import os
import time

# Полное подавление предупреждений
warnings.filterwarnings('ignore')
os.environ['LANGCHAIN_WARNING'] = 'false'
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_VERBOSE'] = 'false'


# Функция загрузки датасета
def load_embeddings_to_dataframe(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return None


# Загрузка данных
df = load_embeddings_to_dataframe("arxiv_embeddings202505211515.csv")


def prepare_documents(df, limit=1000):
    """
    Преобразует DataFrame в список документов LangChain
    """
    documents = []
    df_subset = df.head(limit)

    for idx, row in df_subset.iterrows():
        page_content = ''

        if 'title' in row:
            page_content += f"Название: {row['title']}\n"
        if 'abstract' in row:
            page_content += f"Аннотация: {row['abstract']}\n"
        if 'authors' in row:
            page_content += f"Авторы: {row['authors']}\n"

        metadata = {}
        if 'category' in row:
            metadata['category'] = row['category']
        if 'date' in row:
            metadata['date'] = row['date']
        if 'arxiv_id' in row:
            metadata['arxiv_id'] = row['arxiv_id']

        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))

    print(f"Подготовлено {len(documents)} уникальных документов")
    return documents


# Подготовка документов
documents = prepare_documents(df, limit=1000)

# Конфигурация OpenAI клиента
client = OpenAI(
    api_key="ZmJhMjUwZTItMDg0ZC00N2E3LWIyNDktYjA4MTQyZGFmMGE4.97f6d089a16317c3aa93b365eda739a8",
    base_url="https://foundation-models.api.cloud.ru/v1"
)


class CustomEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов для работы с API"""

    def __init__(self, client, model="BAAI/bge-m3"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для списка документов"""
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.model
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Ошибка при получении эмбеддинга: {e}")
                # Возвращаем нулевой эмбеддинг в случае ошибки
                embeddings.append([0.0] * 1024)  # Зависит от размерности модели
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Получение эмбеддинга для запроса"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Ошибка при получении эмбеддинга запроса: {e}")
            return [0.0] * 1024


# Создание экземпляра эмбеддингов
embeddings = CustomEmbeddings(client)

# Создание векторного хранилища ChromaDB
print("Создание векторного хранилища...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="arxiv_papers",
    persist_directory="./chroma_db"
)
print("Векторное хранилище успешно создано!")

# Создание ретривера
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Функция для безопасного поиска документов
def safe_retrieve(retriever, query):
    """Безопасный поиск документов с обработкой ошибок"""
    try:
        # Используем invoke
        return retriever.invoke(query)
    except Exception as e2:
        print(f"Ошибка при использовании invoke: {e2}")
        return []


# Тестирование ретривера
query = "глубокое обучение для обработки изображений"
retrieved_docs = safe_retrieve(retriever, query)

print(f"Найдено документов: {len(retrieved_docs)}")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nДокумент {i}:")
    print(doc.page_content[:150] + "...")

# MMR балансирует между релевантностью и разнообразием результатов
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Количество документов для первичной выборки
        "lambda_mult": 0.5  # Баланс между релевантностью (1.0) и разнообразием (0.0)
    }
)

# Сравнение результатов
query = "обработка естественного языка"
docs_similarity = safe_retrieve(retriever, query)
docs_mmr = safe_retrieve(retriever_mmr, query)

print("Сравнение результатов поиска:")
print("\n=== Similarity Search ===")
for i, doc in enumerate(docs_similarity[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

print("\n=== MMR Search ===")
for i, doc in enumerate(docs_mmr[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

# Ретривер с фильтрацией по оценке схожести
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Минимальная оценка схожести
        "k": 10
    }
)

def compare_mmr_lambda(query: str, vectorstore, lambdas: List[float] = [0.0, 0.5, 1.0], k: int = 3):
    """
    Сравнивает результаты MMR-поиска при разных значениях lambda_mult.
    """
    print(f"\n Сравнение MMR при разных lambda_mult для запроса: '{query}'\n")

    for lam in lambdas:
        print(f"--- lambda_mult = {lam} ---")
        retriever_mmr = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 20,
                "lambda_mult": lam
            }
        )
        docs = safe_retrieve(retriever_mmr, query)
        for i, doc in enumerate(docs[:k], 1):
            title = doc.page_content.split('\n')[0] if doc.page_content else "Без названия"
            print(f"  {i}. {title[:120]}...")
        print()


def benchmark_retrievers(query: str, retrievers: dict, repeats: int = 3):
    """
    Сравнивает время выполнения разных ретриверов.
    """
    results = {}

    for name, retriever in retrievers.items():
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            docs = safe_retrieve(retriever, query)
            end = time.perf_counter()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        results[name] = {
            "avg_time_sec": avg_time,
            "docs_retrieved": len(docs)
        }

    # Вывод результатов
    print(f"\n Бенчмарк ретриверов для запроса: '{query}' (среднее из {repeats} запусков)\n")
    print(f"{'Ретривер':<25} | {'Время (сек)':<12} | Найдено документов")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<25} | {res['avg_time_sec']:<12.4f} | {res['docs_retrieved']}")
    print()

    return results

# 1. Эксперимент с lambda_mult
compare_mmr_lambda(
    query="машинное обучение в медицине",
    vectorstore=vectorstore,
    lambdas=[0.0, 0.5, 1.0],
    k=3
)

# 2. Сравнение скорости работы ретриверов
retrievers_to_test = {
    "Similarity (k=5)": retriever,
    "MMR (lambda=0.5)": retriever_mmr,
    "Threshold (≥0.7)": retriever_threshold
}

benchmark_retrievers(
    query="нейронные сети для обработки текста",
    retrievers=retrievers_to_test,
    repeats=3
)