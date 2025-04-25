import os

from langchain_community.document_loaders import UnstructuredFileLoader,DirectoryLoader,PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma




# embedding model
embeddings =HuggingFaceEmbeddings()


# Loading documents from a directory
loader = DirectoryLoader(
  path="./data",
  glob="./*.pdf",
  loader_cls=PyPDFLoader
)

documents=loader.load()
print("Total length : ",len(documents))

# text splitting
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

# text chunking 
text_chunks=text_splitter.split_documents(documents)
print("Total length after chunking : ",len(text_chunks)) 

# text chunks storing in vector database
vector_store=Chroma.from_documents(
  documents=text_chunks,
  embedding=embeddings,
  persist_directory="./chroma_db"
)

print("documents stored in vector database")