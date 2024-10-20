import os
import re
import markdown
from pdfminer.high_level import extract_text as extract_text_from_pdf
from io import StringIO
from html.parser import HTMLParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Класс для очистки HTML-тегов из текста
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    """Удалить HTML-теги из строки."""
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def clean_markdown(text):
    """Очистить синтаксис Markdown из текста."""
    # Удалить ссылки в формате Markdown
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Удалить маркеры жирного и курсивного текста
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Удалить изображения и их ссылки
    text = re.sub(r'!\[[^\]]*]\([^)]*\)', '', text)
    # Удалить маркеры заголовков
    text = re.sub(r'#+\s?', '', text)
    # Удалить другой синтаксис Markdown (например, таблицы, маркеры списка)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'\n{2,}', '\n', text)  # Удалить лишние пустые строки
    return text


def extract_text_from_md(md_path):
    """Извлечь и очистить текст из Markdown-файла."""
    with open(md_path, "r", encoding="utf-8") as file:
        md_content = file.read()
        html = markdown.markdown(md_content)
        text = strip_tags(html)
        return clean_markdown(text)


def extract_text_from_file(file_path):
    """Извлечь текст из файла на основе его расширения."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.md'):
        return extract_text_from_md(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return "Неподдерживаемый формат файла."


# Директория, содержащая документы для обработки
directory = r'../data/docs'

# Параметры для разбиения текста
chunk_size = 1200
chunk_overlap = 300

# Список для хранения всех частей документов
all_docs = []
allowed_extensions = ['.md', '.pdf', '.txt']

# Обработка каждого файла в директории
for root, dirs, files in os.walk(directory):
    for filename in files:
        # Получить расширение файла
        _, file_extension = os.path.splitext(filename)
        if file_extension in allowed_extensions:
            file_path = os.path.join(root, filename)  # Полный путь к файлу

            # Удалить расширение ".md", ".pdf" или ".txt" из имени файла
            file_name_without_extension = os.path.splitext(filename)[0]

            # Открыть и прочитать файл
            file_content = extract_text_from_file(file_path)

            # Разбить текст на части
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_text(file_content)

            for i, chunk in enumerate(docs):
                # Определить метаданные для каждой части (можно настроить по своему усмотрению)
                metadata = {
                    "File Name": file_name_without_extension,
                    "Chunk Number": i + 1,
                }

                # Создать заголовок с метаданными и именем файла
                header = f"File Name: {file_name_without_extension}\n"
                for key, value in metadata.items():
                    header += f"{key}: {value}\n"

                # Объединить заголовок, имя файла и содержимое части
                chunk_with_header = header + file_name_without_extension + "\n" + chunk
                all_docs.append(chunk_with_header)

            print(f"Обработано: {filename}")


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# Инициализация HuggingFaceInstructEmbeddings
model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf_embedding = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Встраивание и индексация всех документов с использованием FAISS
db = FAISS.from_texts(all_docs, hf_embedding)

# Сохранение индексаированных данных локально
db.save_local("../data/index/faiss_AiDoc")
