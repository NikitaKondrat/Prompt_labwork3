import sys
import types

# Создаём "виртуальный" модуль langchain с нужными атрибутами для совместимости
if "langchain" not in sys.modules:
    fake_langchain = types.ModuleType("langchain")
    fake_langchain.debug = False
    fake_langchain.llm_cache = None  # ← добавлено!
    sys.modules["langchain"] = fake_langchain

import os
import warnings
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from typing import List

# Импорты для RAG
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Подавление предупреждений
warnings.filterwarnings('ignore')
os.environ['LANGCHAIN_WARNING'] = 'false'
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_VERBOSE'] = 'false'


# === Эмбеддинг-функция (требуется для загрузки Chroma) ===
client = OpenAI(
    api_key="ZmJhMjUwZTItMDg0ZC00N2E3LWIyNDktYjA4MTQyZGFmMGE4.97f6d089a16317c3aa93b365eda739a8",
    base_url="https://foundation-models.api.cloud.ru/v1"
)

class CustomEmbeddings(Embeddings):
    def __init__(self, client, model="BAAI/bge-m3"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(input=[text], model=self.model)
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Ошибка эмбеддинга: {e}")
                embeddings.append([0.0] * 1024)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Ошибка запроса эмбеддинга: {e}")
            return [0.0] * 1024


embeddings = CustomEmbeddings(client)

# === ЗАГРУЗКА СУЩЕСТВУЮЩЕГО ChromaDB ===
print("Загрузка существующего векторного хранилища...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="arxiv_papers"
)
print("Векторное хранилище успешно загружено!")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)


# === RAG: Языковая модель ===
llm = GigaChat(
    credentials="YTViNDBiOGUtNzE2MS00MmQ1LWE5NmYtZjEzOWYwZjQzZjAxOmU3MTQzOGYxLTc4ZWMtNDFkZS05MzkzLWIxNDQ4NThmMThkOQ==",
    scope="GIGACHAT_API_B2B",
    model="GigaChat-2-Max",
    verify_ssl_certs=False,
)


# === Промпт ===
prompt_template = """Ты -- научный ассистент, специализирующийся на анализе научных статей.
Твоя задача -- отвечать на вопросы пользователя, основываясь ТОЛЬКО на предоставленном контексте из научных статей ArXiv.

Правила:
1. Используй только информацию из контекста ниже
2. Если в контексте нет информации для ответа, честно скажи об этом
3. Указывай, из каких статей взята информация (если есть метаданные)
4. Отвечай на русском языке, четко и структурированно
5. Если вопрос касается технических деталей, будь точным

Контекст из научных статей:
{context}

Вопрос пользователя: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(prompt_template)


# === Форматирование контекста ===
def format_docs(docs):
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Документ {i}]")
        context_parts.append(doc.page_content)
        if doc.metadata:  # исправлено: было doc.meta
            context_parts.append(f"Метаданные: {doc.metadata}")
        context_parts.append("")
    return "\n".join(context_parts)


# === RAG-цепочка ===
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


def interactive_rag_qa():
    """Интерактивная система вопросов-ответов"""
    print("=== Интерактивная RAG-система для научных статей ArXiv ===")
    print("Введите 'выход' для завершения\n")

    while True:
        question = input("Ваш вопрос: ").strip()

        if question.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        if not question:
            continue

        try:
            # Получаем релевантные документы
            docs = retriever.invoke(question)
            print(f"\n📚 Найдено релевантных документов: {len(docs)}")

            # Генерируем ответ
            response = rag_chain.invoke(question)
            print(f"\n🤖 Ответ:\n{response.content}\n")

            # Показываем источники
            show_sources = input("Показать источники? (да/нет): ").strip().lower()
            if show_sources in ['да', 'yes', 'y', 'д']:
                print("\n📖 Источники:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
                    print(f"   Метаданные: {doc.metadata}")

            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"❌ Ошибка: {e}\n")


interactive_rag_qa()
