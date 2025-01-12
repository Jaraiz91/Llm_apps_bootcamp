import streamlit as st
import os
import json
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


#Script para configuración del chatbot. El usuario va indicar instrucciones al chatbot sobre como debe de actuar además de cargar documentos que le servirán como referencia a las preguntas que le puedan hacer

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']

USER = 'test'
BASE_PATH = './data/users/'



st.title('Configure su Chatbot')


embeddings = OllamaEmbeddings(model='nomic-embed-text')
splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)


def get_file(key):
    file = st.file_uploader(label="Upload a document", type=['txt', 'pdf'], key=key)
    return file

def load_content(file):
    allowed_files = ['.txt', '.pdf']
    temp_file_path = os.path.join(os.getcwd(), file.name)
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())
    _ ,ext = os.path.splitext(file.name)
    if ext == '.txt':
        loader = TextLoader(temp_file_path)
    elif ext == '.pdf':
        loader = PyPDFLoader(temp_file_path)

    else:
        print(f'File extension not suported. Try uploading one of these ones {allowed_files}', ext)
    docs = loader.load()
    #content = " ".join([x.page_content for x in docs])
    return docs

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

# Necesitamos que responda a una serie de preguntas para generar un prompt que nos sirva como instrucciones iniciales a la hora de cargar el agente

st.write("En primer lugar necesitamos conteste a una serie de preguntas para poder personalizar al agente según sus necesidades:")

# Primera pregunta

st.markdown('## Cuéntanos un poco de ti')
description = st.text_input('Danos una breve descripción de tu negocio')

# Esta pregunta nos sirve para dar instrucciones al LLM sobre como debe expresarse y en que lenguaje hablar. Se puede plantear interpretarlo de la pregunta anterior (descripción)
st.markdown('## Tienes un público objetivo')
public = st.text_input('Puede ser un público en concreto o genérico')

st.markdown('## Instrucciones personalizadas')
instructions = st.text_input('Una breve instrucción sobre que se espera de él. Ej: ayudar a resolver dudas sobre un servicio, gestión de reservas...')

st.markdown('## Documentación')

st.write('Proporciona documentos que sirvan de apoyo para el agente para responder a las preguntas que le hagan. Los documentos pueden contener ifnormaciónd detallada sobre servicios o productos, respuestas a preguntas frecuentes u horarios y disponibilidades en otros ejemplos')




if "file_uploaders" not in st.session_state:
    st.session_state.file_uploaders = [] 

list_of_documents = []
if st.button("Añadir archivo"):
    #print('docs:',len(st.session_state.file_upploaders))
    st.session_state.file_uploaders.append(None)

for i, _ in enumerate(st.session_state.file_uploaders):
    file = get_file(key=f"file_uploader_{i}")
    print(file)
    if file:
        document = load_content(file=file)
        st.session_state.file_uploaders[i] = file

if st.button('Submit'):
    check, needed_elems = check_elements(description=description, public=public, instructions=instructions)
    if not check:
        place_warnings(needed_elems)
        st.stop()
    docs = []
    doc_names = []
    with st.spinner('Uploading...'):
        for f in st.session_state.file_uploaders:
            document = load_content(file=f)
            splitted_doc = splitters.split_documents(documents=document)
            docs += splitted_doc
            doc_names.append(f.name)
        path_to_db = BASE_PATH + os.sep + USER + os.sep + 'test_vectordb'
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=path_to_db)
        vectorstore.persist()
        metadata = {
            USER: {'prompts':{
            'description': description,
            'public': public,
            'instructions': instructions
            },
            'documentos': doc_names
            }
            }
        
        path_to_metadata = BASE_PATH + os.sep + USER + os.sep + 'metadata' + os.sep + 'test.json'
        with open(path_to_metadata, 'w') as jfile:
            json.dump(metadata,jfile, indent=4)







