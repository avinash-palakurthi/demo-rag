# import os 

# import streamlit as st

# from rag import process_document_to_chroma_db, answer_question

# working_dir=os.getcwd()
# print("working : ",working_dir)


# st.title("🤖 RAG-DOC-BOT")

# # file upload
# upload_file=st.file_uploader("Upload a file",type=["pdf"])

# if upload_file is not None:
#   #* save path created
#   save_path=os.path.join(working_dir,upload_file.name)
#   #* saving the file to the path
#   with open(save_path,"wb") as f:
#     f.write(upload_file.getbuffer())
    
#   process_document=process_document_to_chroma_db(upload_file.name)
#   st.info("File uploaded and processed successfully")
  
  
  
# # * user input for asking questions related to the file uploaded
# user_question=st.text_area("Ask a question about the file uploaded")

# if st.button("Answer"):
#   answer=answer_question(user_question)
#   st.markdown("Answer : ")
#   st.markdown(answer)



import os

import streamlit as st

from rag import process_document_to_chroma_db, answer_question


# set the working directory
working_dir = os.getcwd()

st.title("🤖 PDF-Based Question & Answering System")

# file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # define save path
    save_path = os.path.join(working_dir, uploaded_file.name)
    #  save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chroma_db(uploaded_file.name)
    st.info("Document Processed Successfully")


# text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):

    answer = answer_question(user_question)

    st.markdown("### Response:")
    st.markdown(answer)