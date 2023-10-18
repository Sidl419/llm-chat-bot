from fastapi import FastAPI 
import redis
import os
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


REDIS_URL = "redis://localhost:6379/0"


app = FastAPI()
redis_client = redis.Redis.from_url(REDIS_URL) #redis.Redis(host='localhost', port=6379, decode_responses=True)


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="../" + os.getenv("MODEL_FILE"),
    max_tokens=5000,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)

prompt = PromptTemplate(template="""
                        You are a helpful, respectful and honest assistant. Always answer as helpfully
                        as possible, while being safe.  Your answers should not include any harmful,
                        unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
                        that your responses are socially unbiased and positive in nature. If a
                        question does not make any sense, or is not factually coherent, explain why
                        instead of answering something not correct. If you don't know the answer to a
                        question, please don't share false information.
                        
                        Answer to this question: {question}""", input_variables=["question"])



#@app.post("/message")
#def message (user_id: str, message: str): 
user_id = 1 
message = "Hello"
memory = ConversationBufferMemory(chat_memory=RedisChatMessageHistory(url=REDIS_URL, session_id=user_id))
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
ai_message = conversation({"question": message})
print(ai_message["text"])
#return {"message": ai_message["text"]}
