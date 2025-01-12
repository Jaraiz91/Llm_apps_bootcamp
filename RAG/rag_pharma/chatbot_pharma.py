import os
from dotenv import load_dotenv, find_dotenv
import json
from datetime import datetime, timedelta
import random
import pickle
from pydantic import BaseModel
import streamlit as st
from streamlit_chat import message
from langchain.tools import StructuredTool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from utils.google_calendar import get_available_slots, crear_evento, borrar_evento

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']

def get_three_random(lista):
    # Verificar si la lista tiene más de 3 elementos
    if len(lista) > 3:
        # Devolver 3 elementos aleatorios usando random.sample
        return random.sample(lista, 3)
    else:
        # Si la lista tiene 3 o menos elementos, devolver la lista original
        return lista
    



BASE_PROMPT = """Eres un asistente experto en la materia {description}. Debes ser de ayuda en todas las preguntas que te hagan al respecto
.Si te preguntan sobre otro tema distinto no intentes responder la pregunta. Si insiste en hablar de otro tema termina la conversación
Utiliza la documentación aportada al tema para responder las preguntas. Si no sabes no lo inventes. Si se trata de una pregunta relacionada con el tema
pero no lo sabes puedes proponer concertar una llamada. Intenta seguir tambien las instrucciones proporcionadas. Intenta comunicarte como descrito 
en el campo público o como se comunicaría el descrito en dicho campo.

Documentación = {documentation}
Instrucciones = {instructions}
publico = {public}"""

get_available_slots_tool = StructuredTool.from_function(
    func=get_available_slots,
    name='Huecos libres',
    description="Función para encontrar los huecos disponibles en Google Calendar. Seguido de esta función se debe usar la \
    función get_three_random para reducir las opciones a 3"
)



tools = [get_available_slots,get_three_random, crear_evento, borrar_evento]
llm = ChatOpenAI(model='gpt-3.5-turbo-0125').bind_tools(tools=tools)


# Inicializar el historial de chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Almacena respuestas del asistente
if 'past' not in st.session_state:
    st.session_state['past'] = []      # Almacena mensajes del usuario

# Contenedor para el historial del chat
response_container = st.container()
# Contenedor para el input del usuario
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Tú:", key='input', height=100)
        submit_button = st.form_submit_button(label='Enviar')

    if submit_button and user_input:
        # Agregar mensaje del usuario al historial
        st.session_state['past'].append(user_input)
        
        # Simular respuesta (aquí podrías integrar un LLM)
        output = f"Esto es una respuesta a: {user_input}"
        
        # Agregar respuesta al historial
        st.session_state['generated'].append(output)

# Mostrar el historial del chat
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))