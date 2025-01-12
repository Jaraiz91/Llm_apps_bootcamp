import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader

import os, sys
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
print(os.environ['LANGCHAIN_PROJECT'])