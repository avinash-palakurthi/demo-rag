# import os
# import json

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA



# working_dir=os.path.dirname(os.path.abspath((__file__)))
# config_data=json.load(open(f"{working_dir}/config.json"))
# GROQ_API_KEY=config_data["GROQ_API_KEY"]
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# # * loading embedding model
# embedding=HuggingFaceEmbeddings()

# # * loading llm model
# llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",temperature=0)

# #* getting document and storing in DB
# def process_document_to_chroma_db(file_name):
  
#   loader=PyPDFLoader(f"{working_dir}/{file_name}")
#   documents=loader.load()
  
#   #* text splitter
  
#   text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  
#   texts=text_splitter.split_documents(documents)
  
#   vectordb=Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=f"{working_dir}/chroma_db")
  
#   return 0 

# #* answering the question using the vectorstore
# def answer_question(user_question):
#   vectordb=Chroma(
#     persist_directory=f"{working_dir}/chroma_db", 
#     embedding_function=embedding
#     )
  
#   # * retriver
#   retriever=vectordb.as_retriever()
  
#   # * creating chain to answer the question
#   qa_chain=RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#   )
  
#   response=qa_chain({"query":user_question})
#   answer=response["result"]
#   return answer


import os
import json

from langchain_community.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


working_dir = os.path.dirname(os.path.abspath((__file__)))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# loading the embedding model
embedding = HuggingFaceEmbeddings()

# load the llm form groq
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)


def process_document_to_chroma_db(file_name):
    # load the doc using unstructured
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    # splitting te text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/chroma_db"
    )
    return 0


def answer_question(user_question):
    # load the persistent vectordb
    vectordb = Chroma(
        persist_directory=f"{working_dir}/chroma_db",
        embedding_function=embedding
    )
    # retriever
    retriever = vectordb.as_retriever()

    # create a chain to answer user question usinng DeepSeek-R1
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer 