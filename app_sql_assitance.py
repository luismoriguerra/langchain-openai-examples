
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import load_prompt
from pathlib import Path
from sql_execution import execute_df_query


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
root_dir = os.path.dirname(os.path.abspath(__file__))
print(root_dir)
st.title("SQL Assistance")
user_input = st.text_input("Enter your question here")
tab_titles = ["result","query"]
tabs = st.tabs(tab_titles)

prompt_template = load_prompt(root_dir + "/prompts/tpch_prompt.yaml")

llm=OpenAI(temperature=0.9)

sql_generation_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

if user_input:
    sql_query= sql_generation_chain(user_input)
    print(sql_query)
    results = execute_df_query(sql_query['text'])
    
    with tabs[0]:
        st.write(results)
    with tabs[1]:
        st.write(sql_query['text'])
    


