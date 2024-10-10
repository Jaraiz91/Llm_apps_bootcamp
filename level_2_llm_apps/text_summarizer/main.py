import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI



template = ChatPromptTemplate.from_messages(
    [
        ("system", "your task is to make a summarize from the texts that user gives to you. Summary extension should not surpass 200 words. Write the summary in english"),
        ("user", "{user_input}")
    ]
)

def get_llm(api_key):
    llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=api_key)
    return llm

def get_openai_key():
    input_text = st.text_input(label="OpenAI API Key ", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

def get_file():
    file = st.file_uploader(label="Choose a file", type=['txt', 'pdf'])
    print('file:', file)
    return file

st.header("**AI Long Text Summarizer**")

col1, col2 = st.columns(2)

with col1:
    st.write("ChatGPT cannot summarize long texts. Now you can do it with this app")
with col2:
    st.write("contact with jaraiz37@gmail.com for personalized AI projects")

st.markdown('## **Enter Your OpenAI API Key**')

api_key = get_openai_key()


st.markdown('## **Upload the text file you want to summarize**')

warning_api = "Please insert a valid  OpenAI api Key!"

file = get_file()
if file:
    temp_file_path = os.path.join(os.getcwd(), file.name)
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())
    _ ,ext = os.path.splitext(file.name)
    if ext == '.txt':
        loader = TextLoader(temp_file_path)
    elif ext == '.pdf':
        loader = PyPDFLoader(temp_file_path)

    else:
        print('Ext!!!', ext)
    docs = loader.load()
    content = " ".join([x.page_content for x in docs])
    

    if not api_key:
        st.warning(warning_api, icon="⚠️")
        st.stop()

    else:
        try:
            llm = get_llm(api_key=api_key)
        except:
            st.warning(warning_api, icon="⚠️")
            st.stop()

    chain = template | llm | StrOutputParser()
    response= chain.invoke({'user_input': content})

    os.remove(temp_file_path)

    st.write(response)



