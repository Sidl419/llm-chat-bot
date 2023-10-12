from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


loader = DirectoryLoader('../', glob="**/*.md")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    lenght_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_text(loader.load())

embeddings_model = OpenAIEmbeddings(open_api_key="...")
db = Chroma.from_documents(chunks, embeddings_model)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.simularity_search(query)
