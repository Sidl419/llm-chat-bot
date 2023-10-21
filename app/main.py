from fastapi import FastAPI 
import os
import time
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import HumanMessage
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from app.vec_base import VecBase


app = FastAPI()
db = VecBase("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", chunk_size=300, chunk_overlap=50)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=os.getenv("MODEL_FILE"),
    use_mlock=True,
    n_ctx=2048,
    last_n_tokens_size=2048,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
    temperature=0,
    n_threads=6,
    model_kwargs={"keep": -1}
)

prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
"""You are a member of our support team at the Russian Tinkoff Bank, and your job 
is to provide quality service to our customers. Please answer their questions in an 
informative and respectful way, avoiding any harmful or unethical content. Provide 
helpful information, but try to be brief. If you cannot answer a question, please honestly admit your inability. 
You must answer in russian."""
), 
MessagesPlaceholder(variable_name="chat_history"), 
SystemMessagePromptTemplate.from_template("Наша поисковая система выдала следующую информацию по текущей теме: {terms}"),
HumanMessagePromptTemplate.from_template("Представься сотрудником банка. Ответь на этот вопрос на русском языке: {question}?")])

@app.post("/message")
def message (user_id: str, message: str): 
    add_info = db.similarity_search(message, k=3)
    memory = ConversationBufferMemory(memory_key="chat_history", 
                                    input_key="question", 
                                    chat_memory=RedisChatMessageHistory(url="redis://127.0.0.1:6379/0", session_id=user_id, ttl=2), 
                                    return_messages=True)
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    start = time.time()
    ai_message = conversation({"question": message, "terms": add_info})
    duration = time.time() - start
    print(f"Inference time {round(duration // 60)} minutes {round(duration % 60)} seconds")
    return {"message": ai_message["text"]}
