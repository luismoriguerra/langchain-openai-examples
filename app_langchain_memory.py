from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import load_prompt, PromptTemplate
from langchain.memory import ConversationBufferMemory

from pathlib import Path
from sql_execution import execute_df_query


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
root_dir = os.path.dirname(os.path.abspath(__file__))
print(root_dir)

st.title("LangChain Memory demo")
user_input = st.text_input("Enter your message here")

memory  = ConversationBufferMemory()
# memory.chat_memory.add_user_message("hello")
# memory.chat_memory.add_ai_message("how are you?")
# print(memory.load_memory_variables({}))

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']},{'output': message['AI']})

prompt_template = PromptTemplate(
    input_variables=['history','input'],
    template="""
    you are conversational bot. user will communicate with you. maintain formal tone in your response
    conversarion history: 
    {history}
    
    human: {input}
    AI: 
    """
    )

llm = OpenAI(temperature=0)
conversation_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory, verbose=True)

# for debugging
# print(conversation_chain("hello"))
# print(conversation_chain("how is the day today ?"))

if user_input:
    response = conversation_chain(user_input)
    message= {'human': user_input, 'AI': response['text']}
    st.session_state.chat_history.append(message)
    st.write(response)
    with st.expander(label='chat history', expanded=False):
        st.write(st.session_state.chat_history)



