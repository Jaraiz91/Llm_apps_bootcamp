import os
from dotenv import load_dotenv, find_dotenv
import json
from pathlib import Path
from datetime import datetime, timedelta
import random
import pickle
from pydantic import BaseModel
from typing import Literal
import streamlit as st
from streamlit_chat import message
from langchain.tools import StructuredTool, tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from utils.google_calendar import get_available_slots, crear_evento, borrar_evento

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']


json_path = Path('./data/users/test/prompts/test.json')
with open(json_path, 'r') as file:
    data = json.load(file)

description = data['test']['description']
public = data['test']['public']
instructions = data['test']['instructions'] 



@tool
def get_available_slots_tool():
    """Función para encontrar los huecos disponibles en Google Calendar"""
    return get_available_slots()

@tool
def crear_evento_tool(fecha:str):
    """Función para crear un evento en Google Calendar"""
    return crear_evento(fecha)

@tool
def borrar_evento_tool(fecha:str):
    """Función para borrar un evento en Google Calendar"""
    return borrar_evento(fecha)

tools_dict = {
    "get_available_slots_tool": get_available_slots_tool,
    "crear_evento_tool": crear_evento_tool,
    "borrar_evento_tool": borrar_evento_tool
}

tools = [get_available_slots_tool, crear_evento_tool, borrar_evento_tool]
llm = ChatOpenAI(model='gpt-3.5-turbo-0125').bind_tools(tools=tools)
llama = ChatOllama(model='llama3')
embeddings = OllamaEmbeddings(model='nomic-embed-text')

#Cargar el vectorstore
vectorstore = Chroma(
    persist_directory='./data/test/test_vectordb/',
    embedding_function=embeddings
)


class Evento(BaseModel):
    start_time: datetime
    end_time: datetime
    summary: str



RAG_MESSAGE_PROMPT = """Eres un asistente experto en la materia {description}. 
Tienes acceso a las siguientes herramientas que puedes usar cuando sea necesario:
- get_available_slots: Para buscar huecos disponibles en el calendario
- get_three_random: Para seleccionar tres huecos aleatorios de una lista
- crear_evento: Para crear un evento en el calendario
- borrar_evento: Para borrar un evento del calendario

Cuando necesites usar una herramienta, hazlo de forma explícita.
Por ejemplo: "Déjame buscar los huecos disponibles..." y luego usa get_available_slots.

Documentación = {context}
Instrucciones = {instructions}
publico = {public}
pregunta = {question}
Historial de la conversación = {chat_history}"""


intent_prompt_message = """Eres un assistente experto en la materia {description}. Para poder ayudar mejor al usuario primero debes
identificar las intenciones del usuario.La intención debe clasificarse como 'informacion', 'agendar cita', 'cancelar cita', 'ver huecos disponibles' o 'otro'
Es importante que la intención de información sea compatible con la materia {description} de la que eres experto. En caso contrario debería clasificarse como 'otro'
IMPORTANTE: tienes que devolver solamente el intent (una de las 5 opciones), no el resto de la respuesta
IMPORTANTE: Si en chat_history no se ha hablado aun de opciones disponibles de fechas ENTONCES NUNCA PONGAS 'agendar cita' en la respuesta, debes devolver 'ver huecos disponibles'
"""

ver_huecos_disponibles_prompt_message = """Dada una lista de dattimes con los huecos disponibles en {result}, devuelvelos en un formato legible para el usuario. """
agendar_cita_prompt_message = """utilizando {questtion} del usuario y {chat_history} debes averiguar la fecha y hora que quiere agendar el usuario y devolverlo en el siguiente formato: "{"start_time:datetime, end_time:datetime, summary:str}" """"
cancelar_cita_prompt_message = """Intenta cancelar una cita en el calendario. Para ello debes usar la herramienta borrar_evento_tool"""
otro_prompt_message = """Deja claro al usuario que no puedes responder a su pregunta ya que solo estás para tratar temas sobre {description}"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_MESSAGE_PROMPT),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{question}")
])

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", intent_prompt_message),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{question}")
])

huecos_disponibles_prompt = ChatPromptTemplate.from_messages([
    ("system", ver_huecos_disponibles_prompt_message),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{question}")
])

agendar_cita_prompt = ChatPromptTemplate.from_messages([
    ("system", agendar_cita_prompt_message),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{question}")
])

cancelar_cita_prompt = ChatPromptTemplate.from_messages([
    ("system", cancelar_cita_prompt_message),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{question}")
])  

otro_prompt = ChatPromptTemplate.from_messages([
    ("system", otro_prompt_message),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{question}")
])  





tools = [get_available_slots_tool, crear_evento_tool, borrar_evento_tool]
llm = ChatOpenAI(model='gpt-3.5-turbo-0125').bind_tools(tools=tools)
embeddings = OllamaEmbeddings(model='nomic-embed-text')

#Cargar el vectorstore
vectorstore = Chroma(
    persist_directory='./data/test/test_vectordb/',
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':3})
#crear la memoria
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)  


    
def debug_print(x):
    print(f"DEBUG RAG Chain - Input recibido: {x}")
    return x

rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | RunnableLambda(lambda x: print(f"DEBUG - Documentos recuperados: {x}") or x) | format_docs,
        "question": RunnablePassthrough.assign(question=lambda x: print(f"DEBUG - Pregunta: {x['question']}") or x["question"]),
        "chat_history": lambda x: print(f"DEBUG - Chat history: {memory.load_memory_variables({})['chat_history']}") or memory.load_memory_variables({})["chat_history"]
    }
    | RunnablePassthrough.assign(
        description=lambda _: print(f"DEBUG - Description: {description}") or description,
        public=lambda _: print(f"DEBUG - Public: {public}") or public,
        instructions=lambda _: print(f"DEBUG - Instructions: {instructions}") or instructions
    )
    | RunnableLambda(debug_print)  # Añadimos un print antes del prompt
    | rag_prompt
    | RunnableLambda(lambda x: print(f"DEBUG - Prompt final: {x}") or x)  # Print del prompt final
    | llm
    | RunnableLambda(lambda x: print(f"DEBUG - Respuesta del LLM: {x}") or x)  # Print de la respuesta
    | StrOutputParser()
)

huecos_disponibles_chain = (
    {"question": RunnablePassthrough.assign(question=lambda x: x["question"]),
    "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
    "result": lambda _ : get_available_slots()
    }
    | huecos_disponibles_prompt
    | llama

)


agendar_cita_chain = (
    {"question": RunnablePassthrough.assign(question=lambda x: x["question"]),
    "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
    }
    | agendar_cita_prompt
    | llm

)   

borrar_cita_chain = (
    {"question": RunnablePassthrough.assign(question=lambda x: x["question"]),
    "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
    }
    | cancelar_cita_prompt
    | llm
    | StrOutputParser()
)  

otro_chain = (
    {"question": RunnablePassthrough.assign(question=lambda x: x["question"]),
    "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
    }| RunnablePassthrough.assign(
        description=lambda _: description)
    | otro_prompt
    | llm
    | StrOutputParser()
)   

intent_chain = (
    {"question": RunnablePassthrough.assign(question=lambda x: x["question"]),
    "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
    }
    | RunnablePassthrough.assign(
        description=lambda _: description,
        public=lambda _: public,
        instructions=lambda _: instructions
    )
    | intent_prompt
    | llm
    | StrOutputParser()
)


def route(info):
    if info['intent'] == 'informacion':
        return rag_chain
    elif info['intent'] == 'agendar cita':
        return agendar_cita_chain
    elif info['intent'] == 'cancelar cita':
        return cancelar_cita_chain
    elif info['intent'] == 'ver huecos disponibles':
        return huecos_disponibles_chain
    else:
        return otro_chain
    nte
full_chain = {'intent': intent_chain , 'question': lambda x: x['question']} | RunnableLambda(route)



# Inicializar la memoria en el estado de la sesión si no existe
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

def process_user_input(user_input: str) -> str:
    try:
        print('DEBUG - Antes de invoke - Memory:', st.session_state.memory.load_memory_variables({}))
        
        # Invocar la cadena con el input del usuario
        output = full_chain.invoke({"question": user_input})
        print('DEBUG - Output recibido:', output)
        
        # Convertir el output a string
        if hasattr(output, 'content'):
            output_str = output.content
        else:
            output_str = str(output)
        
        # Guardar el contexto en la memoria de la sesión
        st.session_state.memory.save_context(
            inputs={"input": str(user_input)},
            outputs={"output": output_str}
        )
        
        print('DEBUG - Después de save_context - Memory:', st.session_state.memory.load_memory_variables({}))
        print('memory', st.session_state.memory.load_memory_variables({})["chat_history"])
        
        return output_str
        
    except Exception as e:
        print(f"Error en process_user_input: {str(e)}")
        return "Lo siento, ocurrió un error al procesar tu pregunta. Por favor, inténtalo de nuevo."



# Inicializar el historial de chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Lista de strings
if 'past' not in st.session_state:
    st.session_state['past'] = []      # Lista de strings

# Contenedor para el historial del chat
response_container = st.container()
# Contenedor para el input del usuario
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Tú:", key='input', height=100)
        submit_button = st.form_submit_button(label='Enviar')

    if submit_button and user_input:
        output_str = process_user_input(user_input)  # Ya recibimos un string
        print('memory', st.session_state.memory.load_memory_variables({})["chat_history"])
        
        # Agregar strings al historial
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output_str)

# Mostrar el historial del chat
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(str(st.session_state["past"][i]), is_user=True, key=str(i) + '_user')
            message(str(st.session_state["generated"][i]), key=str(i))