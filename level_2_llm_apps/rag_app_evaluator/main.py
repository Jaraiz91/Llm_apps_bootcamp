import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.evaluation.qa import QAEvalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']
groq_api_key = os.environ['GROQ_API_KEY']
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
huggingface_embedding_model_id = 'sentence-transformers/all-MiniLM-L6-v2'


st.markdown('# **RAG App evaluator**')

template = """Your task is to answer the questions you are given by the user.
    Try to use only the context given to answer. In case you don´t find the answer try not to make it up. 
    Use strictly the context given. Try also to answer the same language the question is made.
    context: {context}"""

rag_prompt = ChatPromptTemplate.from_messages([
    ('system', template),
    ('human', '{input}')
])

def load_content(file):
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
    #content = " ".join([x.page_content for x in docs])
    return docs

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
    return file

def check_elements(**kwargs):
    need_elems = []
    for i in kwargs:
        if kwargs[i] == None or kwargs[i]=='':
            need_elems.append(i)
        else:
            continue
    if len(need_elems) > 0:
        check = False
    else:
        check = True
    return check, need_elems

def place_warnings(elems):
    for e in elems:
        st.warning(f'{e} must be provided', icon="⚠️")
    return


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
    check, needed_elems = check_elements(file=file, question=question, answer=answer)
    if not check:
        place_warnings(needed_elems)
        st.stop()
    real_qa = [
        {
            "question": question,
            "answer": answer
        }
    ]
    with st.spinner('Evaluating responses...'):
        openai_llm = get_llm(model='openai')
        llama_llm = get_llm(model='llama3.1')
        qa_chain_llama = create_stuff_documents_chain(llm=llama_llm, prompt=rag_prompt)
        qa_chain_openai = create_stuff_documents_chain(llm=openai_llm, prompt=rag_prompt)
        docs = load_content(file)
        splits = text_splitter.split_documents(docs)
        try:
            print('chromaaaaa')
            OpenAi_vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            Llama_vectorstore = FAISS.from_documents(documents=splits, embedding=HuggingFaceBgeEmbeddings(model_name=huggingface_embedding_model_id))
        except Exception as e:
            print('error')
            st.error(f"Error al conectar con FAISS: {e}")
            st.stop()
        OpenAi_retriever = OpenAi_vectorstore.as_retriever()
        Llama_retriever = Llama_vectorstore.as_retriever()
        openai_retrieval_chain = create_retrieval_chain(OpenAi_retriever, qa_chain_openai) 
        llama_retrieval_chain = create_retrieval_chain(Llama_retriever, qa_chain_llama)

        llama_response = llama_retrieval_chain.invoke({'input': question})['answer']
        openai_response = [openai_retrieval_chain.invoke({'input': question})]
        qa_pairs = {'question': question, 'answer': answer}
        openai_evaluator = QAEvalChain.from_llm(openai_llm)
        llama_evaluator = QAEvalChain.from_llm(llama_llm)
        openai_eval = openai_evaluator.evaluate(
            real_qa,
            openai_response,
            question_key='question',
            prediction_key='answer',
            answer_key='answer'
        )
    st.write(openai_eval)
    st.divider()
    st.write('All done!', openai_response)
    st.write('llama:', llama_response)



