import streamlit as st
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.evaluation.qa import QAEvalChain
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']
groq_api_key = os.environ['GROQ_API_KEY']


st.markdown('# **RAG App evaluator**')

prompt = """Your task is to answer the questions you are given by the user.
    Try to use only the context given to answer. In case you don´t find the answer try not to make it up. 
    Use strictly the context given. Try also to answer the same language the question is made.
    context: {context}"""

def get_llm(model):
    if model == 'openai':
        llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)
    elif model == 'llama3.1':
        llm = ChatGroq(model='llama-3.1-70b-versatile')
    return llm

def evaluate_qa(question, answer, llm):
    evaluator = QAEvalChain.from_llm(llm)
    qa_pairs = [{'question': question, 'answer': answer}]
    result = evaluator.evaluate(qa_pairs)[0]
    return result['answer_generated'], result['is_correct'], result['feedback']

def get_file():
    file = st.file_uploader(label="Upload a document", type=['txt', 'pdf'])
    print('file:', file)
    return file

def check_elements(**kwargs):
    if None in kwargs.values():
        return False
    else:
        return True

with st.expander('Evaluate the quality of a RAG App'):
    st.write("""
            To evauluate the quality of a RAG app, we will ask it questions for which we already know the real
             answers.

             That way we can see if the app i producing the right answers or if it is hallucinating.
             """)
    

file = get_file()

question = st.text_input(label='Enter a question you have already fact checked')
answer = st.text_input(label='Enter the real answer to the question')


button = st.button('Submit', use_container_width=True)

if button:
    with st.spinner('Evaluating responses...'):
        openai_llm = get_llm(model='openai')
        llama_llm = get_llm(model='llama3.1')
