import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ast
import matplotlib.pyplot as plt


# Функция загрузки датасета
def load_embeddings_to_dataframe(filepath):
    """
    Загружает данные эмбеддингов из CSV файла в pandas DataFrame

    Args:
        filepath (str): Путь к файлу с данными

    Returns:
        pd.DataFrame: DataFrame с загруженными данными
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Загружено {len(df)} записей из файла {filepath}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"\nПервые 3 записи:")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return None


# Загрузка данных
df = load_embeddings_to_dataframe("arxiv_embeddings202505211515.csv")


def prepare_documents(df, limit=1000):
    """
    Преобразует DataFrame в список документов LangChain

    Args:
        df: DataFrame с данными статей
        limit: Количество документов для обработки

    Returns:
        List[Document]: Список документов
    """
    documents = []

    # Возьмите первые limit записей
    df_subset = df.head(limit)


    for idx, row in df_subset.iterrows():
        page_content = ''
        # Формируйте текст документа из доступных полей

        # Адаптируйте под структуру вашего датасета
        if 'title' in row:
            page_content += f"Название: {row['title']}\n"
        if 'abstract' in row:
            page_content += f"Аннотация: {row['abstract']}\n"
        if 'authors' in row:
            page_content += f"Авторы: {row['authors']}\n"

        # Метаданные
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

# Просмотр примера документа
print("Пример документа:")
print(f"Содержимое: {documents[0].page_content[:200]}...")
print(f"Метаданные: {documents[0].metadata}")

# Конфигурация OpenAI клиента для получения эмбеддингов
client = OpenAI(
api_key="ZmJhMjUwZTItMDg0ZC00N2E3LWIyNDktYjA4MTQyZGFmMGE4.97f6d089a16317c3aa93b365eda739a8",
    base_url="https://foundation-models.api.cloud.ru/v1"
)


def get_embedding(text: str, model="BAAI/bge-m3") -> list:
    """Получает эмбеддинг текста"""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Тестирование функции эмбеддингов
test_embedding = get_embedding("Тестовый запрос")
print(f"Размерность эмбеддинга: {len(test_embedding)}")

from langchain_core.embeddings import Embeddings
from typing import List


class CustomEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов для работы с API"""

    def __init__(self, client, model="BAAI/bge-m3"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для списка документов"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Получение эмбеддинга для запроса"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding


# Создание экземпляра эмбеддингов
embeddings = CustomEmbeddings(client)

# Создание векторного хранилища ChromaDB
print("Создание векторного хранилища...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="arxiv_papers",
    persist_directory="./chroma_db"  # Директория для сохранения
)
print("Векторное хранилище успешно создано!")

# Проверка работы хранилища
test_query = "нейроные сети"
results = vectorstore.similarity_search_with_score(query=test_query, k=5)
print(f"\nРезультаты поиска по запросу '{test_query}':")

for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Схожесть: {score:.4f}")
    print(f"   Документ: {doc.page_content[:150]}...")
    print(f"   Метаданные: {doc.metadata}")
    print()