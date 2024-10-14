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
    loader = WebBaseLoader(url)
    docs = loader.load()
    content = ' '.join([x.page_content for x in docs])
    return content

classify_prompt_template = """Your task is to extract the information from the following passage.
Use the format provided for the output. Provide the answer in english
passge:
{input}"""

classify_prompt = ChatPromptTemplate.from_template(classify_prompt_template)
summary_prompt_template = """Your task is to provide a summary in English no longer than 50 words to explain
{input}"""
summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)
st.markdown('# **Key data extractor**')

def get_llm(model,api_key, schema=False):
    if model == 'gpt-3.5':
        if schema == True:
            llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=api_key).with_structured_output(schema=PageClassifier)
        else:
            llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=api_key)
    else:
        if schema == True:
            llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=api_key).with_structured_output(schema=PageClassifier)
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

col1, col2 = st.columns(2)

with col1:
    reset = st.button('reset', use_container_width=True)
with col2:
    button = st.button('Analyze', use_container_width=True)

if reset:
    st.stop()

if button:
    warning_api = "Please insert a valid api Key!"
    if url:
        # Check for the Api Key
        if api_key:
            try:
                llm_with_schema = get_llm(model=model,api_key=api_key, schema=True)
                llm = get_llm(model=model,api_key=api_key)

            except:
                st.warning(warning_api, icon="⚠️")
                st.stop()
        else:
            st.warning(warning_api, icon="⚠️")
            st.stop()
        # checking the url
        try:
            content = load_page_content(url=url)
        except:
            not_valid_url = 'Seems the url provided is not valid, please try again with another one'
            st.warning(not_valid_url, icon="⚠️")


        classify_chain = classify_prompt | llm_with_schema
        classified_url = classify_chain.invoke({'input': content})
        summary_chain = summary_prompt | llm | StrOutputParser()
        summary_response = summary_chain.invoke({'input': content})

        topic, sentiment, language = st.columns(3)
        with topic:
            st.markdown('### **Topic:**')
            st.write(classified_url.topic)
        with sentiment:
            st.markdown('### **Sentiment:**')
            st.write(classified_url.sentiment)
        with language:
            st.markdown('### **Language:**')
            st.write(classified_url.language)
        
        st.markdown('## **Summary:**')
        st.write(summary_response)


    else:
        not_url = 'Please provide a valid url'
        st.warning(not_url, icon="⚠️")



        

        
