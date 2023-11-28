
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain.llms import OpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

st.title("LangChain")
prompt = st.text_input("enter your prompt")

llm = OpenAI(temperature=0.9)

if prompt:
    response = llm(prompt=prompt)
    st.write(response)