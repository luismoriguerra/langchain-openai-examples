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
from langchain import HuggingFaceHub


HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

template = """ Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = "google/flan-t5-xxl"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 128}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "what is the capital of india?"
result = llm_chain(question)
print(result["text"])
