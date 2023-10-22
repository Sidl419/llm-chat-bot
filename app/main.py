from fastapi import FastAPI
import os
import time
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from app.vec_base import VecBase


app = FastAPI()
db = VecBase(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size=300,
    chunk_overlap=50,
)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=os.getenv("MODEL_FILE"),
    use_mlock=True,
    n_ctx=2048,
    last_n_tokens_size=2048,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    temperature=0,
    n_threads=6,
    model_kwargs={"keep": -1},
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """Вы являетесь ценным сотрудником службы поддержки Тинькофф Банка, Россия. Ваша работа заключается в 
            предоставлении отличного сервиса и поддержки нашим клиентам. Мы ожидаем, что вы будете отвечать на 
            их запросы в ясной, информативной и уважительной манере. Мы просим вас избегать любой вредной, 
            неэтичной или предвзятой информации. Если вы не можете ответить на вопрос клиента, пожалуйста, 
            объясните причину вашего незнания. Как команда, мы стремимся поддерживать позитивный имидж бренда и 
            относимся друг к другу с уважением. Пожалуйста, будьте кратки и предлагайте ответы на русском языке."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        SystemMessagePromptTemplate.from_template(
            "Наша поисковая система выдала следующую информацию по текущей теме: {terms}"
        ),
        HumanMessagePromptTemplate.from_template(
            "Представься сотрудником банка. Ответь на этот вопрос на русском языке: {question}?"
        ),
    ]
)


@app.post("/message")
def message(user_id: str, message: str):
    add_info = db.similarity_search(message, k=3)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        chat_memory=RedisChatMessageHistory(url="redis://127.0.0.1:6379/0", session_id=user_id, ttl=2),
        return_messages=True,
    )
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    start = time.time()
    ai_message = conversation({"question": message, "terms": add_info})
    duration = time.time() - start
    print(f"Inference time {round(duration // 60)} minutes {round(duration % 60)} seconds")
    return {"message": ai_message["text"]}
