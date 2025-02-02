import os
from dotenv import load_dotenv, find_dotenv
import json
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


from utils.google_calendar import get_available_slots, crear_evento, borrar_evento

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']
client_id = os.environ['GOOGLE_CLIENT_ID']
client_secret = os.environ['GOOGLE_CLIENT_SECRET']


BASE_MESSAGE_PROMPT = """Eres un asistente experto en la materia {description}. 
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

embeddings = OllamaEmbeddings(model='nomic-embed-text')

#cargar el vectorstore
vectorstore = Chroma(
    persist_directory='./data/test/test_vectordb/',
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':3})
llm = ChatOpenAI(model='gpt-3.5-turbo')

class ChatBotState(MessagesState):
    Intention : str
    on_calendar_node : bool = False

class CalendarState(MessagesState):
    Preferred_date: str = None
    on_calendar_node : bool = False

class Intention(BaseModel):
    intention : Literal["Informacion", "calendario", "otro"] = Field(description="La intención que se saca del mensaje que da el usuario humano. Se tiene que clasificar entre información o calendario")
    

class Calendarday(BaseModel):
    day : datetime = Field(description="La fecha para agendar una cita")

INTENT_BASE_MESSAGE = """Eres un asistente en de una farmacia y tu objetivo es ayudar a los clientes a despejar sus dudas. Entre tus funciones está resolver dudas relacionadas con el tema, así como gestionar citas para una reunión en persona con uno de nuestros farmaceúticos. Para poder ayudar mejor, tu tarea va a ser clasificar la intención del usuario entre las \
siguientes opciones : \
   1. Información -> en caso de que necesite saber u obtener consejo sobre la farmacia. Por ejemplo: el usuario quiere saber que dias hacemos guardias \
   2. Calendario: El usuario quiere agendar una cita \
   3. Otro: Esta hablando sobre un tema que no está relacionada con la farmacia."""


def get_preferred_day(state: )

def get_user_intention(state: ChatBotState):
    llm_with_structured_output = llm.with_structured_output(Intention)
    response = llm_with_structured_output.invoke([AIMessage(INTENT_BASE_MESSAGE)] + state['messages'])
    return {'messages': AIMessage(response.intention), 'intention':response.intention}

def intention_router(state: ChatBotState):
    intent = state['intention']
    return intent


