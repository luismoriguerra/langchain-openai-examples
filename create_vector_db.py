
from dotenv import load_dotenv
load_dotenv()

import os
from langchain.embeddings import OpenAIEmbeddings
import tiktoken
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_token_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

doc_path = f'{root_path}/transcripts'
db_path = f'{root_path}/vector_db'

doc_loader = DirectoryLoader(path=doc_path, glob='*.txt', loader_cls=TextLoader)

docs = doc_loader.load()
print(f'Loaded {len(docs)} documents')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_token_len,
    separators=['\n\n','\n', ' ','']
    
)

chunks = text_splitter.split_documents(documents=docs)
print(f'Created {len(chunks)} chunks')
print(chunks[0])


vector_db = FAISS.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
vector_db.save_local(db_path)