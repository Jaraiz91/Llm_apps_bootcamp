
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def get_llm(openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    return llm

def get_openai_key():
    input_text = st.text_input(label="OpenAI API Key ", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text


with st.sidebar:
    llm = get_openai_key()


st.header('**BlogPost generator**')