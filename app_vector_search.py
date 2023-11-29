from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import load_prompt, PromptTemplate
from langchain.memory import ConversationBufferMemory


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
root_path = os.path.dirname(os.path.abspath(__file__))
db_path = f'{root_path}/vector_db'
print(root_path)

st.title("Vector DB demo")
user_prompt = st.text_input("enter your question here")

vector_db = FAISS.load_local(folder_path=db_path, embeddings=OpenAIEmbeddings())
prompt_template = load_prompt(f'{root_path}/prompts/llm_prompt.json')

llm = OpenAI()

if user_prompt:
    docs = vector_db.similarity_search(user_prompt)
    final_prompt = prompt_template.format(docs=docs, user_prompt=user_prompt)
    response = llm.predict(text=final_prompt)
    st.write(response)
    with st.expander(label='docs', expanded=False):
        st.write(docs)