import os
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings  #* for embedding questions
from langchain_chroma import Chroma  #* for vector database
from langchain_groq import ChatGroq  #* for LLM model
from langchain.memory import ConversationBufferMemory  #* for memory
# from langchain.chains import ConversationalRetrivalChain
from langchain.chains import RetrievalQA

# load_dotenv()

# groq_api=os.getenv("GROQ_API_KEY")

# print(groq_api)

#* Set the working directory to the directory of this file
working_dir=os.path.dirname(os.path.abspath(__file__))

config_data=json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY=config_data["GROQ_API_KEY"]

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def vectorstore():
  
  persistence_dir = f"{working_dir}/chroma_db"
  embeddings=HuggingFaceEmbeddings()
  vectorstore=Chroma(persist_directory=persistence_dir, embedding_function=embeddings)
  
  return vectorstore


  