
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser


prompt_template = """
    You are an assistant whose task is to generate a post content based on the topic given by the user input.
    You will generate the content in a format apropiate for social network that is specified:
    topic : {user_topic}
    social network : {user_site}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
parser = StrOutputParser()


def get_llm(openai_api_key):
    llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-3.5-turbo')
    return llm

def get_openai_key():
    input_text = st.text_input(label="OpenAI API Key ", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text


with st.sidebar:
    api_key = get_openai_key()
    if api_key:
        llm = get_llm(openai_api_key=api_key)


st.header('**BlogPost generator**')


topic = st.text_input(label="Post topic", placeholder="Introduce a topic")


site = st.selectbox(label="select a site",options=("Instasgram", "LinkedIn", "Facebook", "Reddit"))
warning_api = "Please insert a valid  OpenAI api Key!"

if topic:
    if api_key:
        try:
            llm = get_llm(openai_api_key=api_key)

        except:
            st.warning(warning_api, icon=":material/warning")
            st.stop()
        chain = prompt | llm | parser
        post = chain.invoke(
            {'user_topic': topic,
                "user_site": site}
        )

        st.write(post)
    else:
        st.warning(warning_api, icon=":material/warning")
        st.stop()

