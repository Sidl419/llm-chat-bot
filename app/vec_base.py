from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.document_loaders.csv_loader import CSVLoader
import os
import time

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    is_separator_regex=False,
    length_function=len
)
embeddings_model =  LlamaCppEmbeddings(model_path="../" + os.getenv("MODEL_FILE"),)

csv_loader = DirectoryLoader('../tinkoff-terms', glob="**/*.csv", loader_cls=CSVLoader)
csv_chunks = text_splitter.split_documents(csv_loader.load())

pdf_loader = DirectoryLoader('../tinkoff-terms', glob="**/*.pdf", loader_cls=PDFMinerLoader)
pdf_chunks = text_splitter.split_documents(pdf_loader.load())

start = time.time()
db = Chroma.from_documents(pdf_chunks + csv_chunks, embeddings_model, persist_directory="./chroma_db")
end = time.time()
print(f"time for db creation: {end - start // 60} minutes")

query = "хочу кредит"
#docs = db.similarity_search(query)
print(db.similarity_search(query))
