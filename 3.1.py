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

subset_df = df.sample(n=1000, random_state=42)
category_counts = subset_df['categories'].value_counts()
top_10_categories = category_counts.head(10)
other_categories = category_counts.iloc[10:]
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_10_categories)), top_10_categories.values)
plt.yticks(range(len(top_10_categories)), top_10_categories.index)
plt.ylabel('Категории статей')
plt.xlabel('Количество статей')
plt.title('Топ-10 категорий по количеству статей')
plt.tight_layout()
plt.show()
print("Остальные категории (от наибольшего к наименьшему):\n")
for category, count in other_categories.items():
    print(f"{category}: {count} статей")