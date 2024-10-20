from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
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
db = FAISS.load_local("../data/index/faiss_AiDoc", hf_embedding)

# Шаблон для вопросно-ответного запроса
template = """Вопрос: {question} \n\nОтвет:"""
# Инициализация шаблона запроса и менеджера обратных вызовов
prompt = PromptTemplate(template=template, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Локальный путь к загруженной модели Llama2
model_path = "../data/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf"

# Инициализация модели LlamaCpp
llm = LlamaCpp(model_path=model_path, temperature=0.2, max_tokens=4095, top_p=1, callback_manager=callback_manager,
               n_ctx=6000)

# Создание LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Определение запроса для поиска в проиндексированных документах
query = "&amp;lt;&amp;lt;что такое латеральное мышление?&amp;gt;&amp;gt;?"
# Поиск семантически похожих фрагментов и возвращение топ-5 фрагментов
search = db.similarity_search(query, k=5)

# Шаблон для генерации итогового запроса
template = '''Контекст: {context}
Исходя из контекста, ответьте на следующий вопрос:
Вопрос: {question}
Предоставьте ответ только на основе предоставленного контекста, без использования общих знаний. Ответ должен быть непосредственно взят из предоставленного контекста.
Пожалуйста, исправьте грамматические ошибки для улучшения читаемости.
Если в контексте нет информации, достаточной для ответа на вопрос, укажите, что ответ отсутствует в данном контексте.
Пожалуйста, включите источник информации в качестве ссылки, поясняющей, как вы пришли к своему ответу.'''

# Создание шаблона для финального запроса
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Форматирование итогового запроса с учетом вопроса и результатов поиска
final_prompt = prompt.format(question=query, context=search)

# Запуск LLMChain для генерации ответа на основе контекста
llm_chain.run(final_prompt)
