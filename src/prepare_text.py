import os
import re
from html.parser import HTMLParser
from io import StringIO

import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text as extract_text_from_pdf

from src.constants import (ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE,
                           DIRECTORY, REGEX_BOLD_AND_ITALIC_LIST,
                           REGEX_HEADERS, REGEX_IMAGES, REGEX_LINKS)


class MLStripper(HTMLParser):
    """Класс для очистки HTML-тегов из текста."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d) -> None:
        self.text.write(d)

    def get_data(self) -> str:
        return self.text.getvalue()


def strip_tags(html: str) -> str:
    """
    Удаляет HTML-теги из строки.

    Args:
        html: Текст HTML.
    """
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def clean_markdown(text: str) -> str:
    """
    Очищает синтаксис Markdown из текста.

    Args:
        text: Текст, который необходимо очистить.
    """
    for regex_string in [REGEX_LINKS, *REGEX_BOLD_AND_ITALIC_LIST]:
        text = re.sub(regex_string, r"\1", text)

    for regex_string in [REGEX_IMAGES, REGEX_HEADERS]:
        text = re.sub(regex_string, "", text)

    # Удалить другой синтаксис Markdown (например, таблицы, маркеры списка)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"-{2,}", "", text)
    text = re.sub(r"\n{2,}", "\n", text)  # Удалить лишние пустые строки
    return text


def extract_text_from_md(md_path: str) -> str:
    """
    Извлекает и очищает текст из Markdown-файла.

    Args:
        md_path: Путь к Markdown-файлу.
    """
    with open(md_path, "r", encoding="utf-8") as file:
        md_content = file.read()
        html = markdown.markdown(md_content)
        text = strip_tags(html)
        return clean_markdown(text)


def extract_text_from_file(file_path: str) -> str:
    """
    Извлекает текст из файла на основе его расширения.

    Args:
        file_path: Путь к файлу.
    """
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".md"):
        return extract_text_from_md(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        return "Неподдерживаемый формат файла."


# Список для хранения всех частей документов
all_docs = []

# Обработка каждого файла в директории
for root, dirs, files in os.walk(DIRECTORY):
    for filename in files:
        # Получить расширение файла
        _, file_extension = os.path.splitext(filename)
        if file_extension in ALLOWED_EXTENSIONS:
            file_path = os.path.join(root, filename)  # Полный путь к файлу

            # Удалить расширение ".md", ".pdf" или ".txt" из имени файла
            file_name_without_extension = os.path.splitext(filename)[0]

            # Открыть и прочитать файл
            file_content = extract_text_from_file(file_path)

            # Разбить текст на части
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            docs = text_splitter.split_text(file_content)

            for index, chunk in enumerate(docs):
                # Определить метаданные для каждой части (можно настроить по своему усмотрению)
                metadata = {
                    "File Name": file_name_without_extension,
                    "Chunk Number": index + 1,
                }

                # Создать заголовок с метаданными и именем файла
                header = f"File Name: {file_name_without_extension}\n"
                for key, value in metadata.items():
                    header += f"{key}: {value}\n"

                # Объединить заголовок, имя файла и содержимое части
                chunk_with_header = header + file_name_without_extension + "\n" + chunk
                all_docs.append(chunk_with_header)

            print(f"Обработано: {filename}")


from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

# Инициализация HuggingFaceInstructEmbeddings
model_name = "hkunlp/instructor-large"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf_embedding = HuggingFaceInstructEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Встраивание и индексация всех документов с использованием FAISS
db = FAISS.from_texts(all_docs, hf_embedding)

# Сохранение индексированных данных локально
db.save_local("../data/index/faiss_AiDoc")
