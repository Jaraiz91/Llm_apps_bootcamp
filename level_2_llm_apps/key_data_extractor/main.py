import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq


class PageClassifier(BaseModel):
    topic : str = Field(description='The main topic the text talks about')
    sentiment : str = Field(description='The sentiment analysis of the text')
    language : str = Field(description='The language the text is written with')  

def load_page_content(url):
    loader = WebBaseLoader()
    docs = loader.load()
    content = ' '.join([x.page_content for x in docs])
    return content

st.markdown('# **Key data extractor**')

def get_llm(model,api_key):
    if model == 'gpt-3.5':
        llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=api_key)
    else:
        llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=api_key)
    return llm

col1, col2 = st.columns(2)

with col1:
    model = st.selectbox(options=['gpt-3.5', 'llama-3.1'],label='model')
with col2:
    api_key = st.text_input(label='api key', type='password')

st.markdown('Provide a url to be analyzed:')

url = st.text_input(label='url:')
