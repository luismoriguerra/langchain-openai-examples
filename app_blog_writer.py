
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from langchain.prompts import load_prompt
from pathlib import Path

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
root_dir = [ p for p in Path(__file__).parents if p.parts[-1] == 'yt-samples'][0]

st.title("LangChain AI blog assistant")
topic = st.text_input("enter your blog topic here")

title_prompt_template = load_prompt(f'{root_dir}/prompts/title_prompt.yaml')
script_prompt_template = load_prompt(f'{root_dir}/prompts/script_prompt.yaml')

llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_prompt_template, output_key="title",verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_prompt_template, output_key="script",verbose=True)

chain_seq=SequentialChain(
    chains=[title_chain,script_chain],
    input_variables=["topic"],
    output_variables=["title","script"],
    verbose=True
)

if topic:
    response = chain_seq({"topic":topic})
    with st.expander(label="title",expanded=False):
        st.write(response["title"])
    with st.expander(label="script",expanded=False):
        st.write(response["script"])
    
    