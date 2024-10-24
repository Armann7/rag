REGEX_LINKS = r"\[([^\]]+)\]\([^)]+\)"
REGEX_BOLD_AND_ITALIC_LIST = [
    r"\*\*([^*]+)\*\*",
    r"\*([^*]+)\*",
    r"__([^_]+)__",
    r"_([^_]+)_",
]
REGEX_IMAGES = r"!\[[^\]]*]\([^)]*\)"
REGEX_HEADERS = r"#+\s?"
# Директория, содержащая документы для обработки
DIRECTORY = r"../data/docs"
# Параметры для разбиения текста
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
ALLOWED_EXTENSIONS = frozenset({".md", ".pdf", ".txt"})
