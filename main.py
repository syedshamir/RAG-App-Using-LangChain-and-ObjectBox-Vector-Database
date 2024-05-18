import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain #to invoke the entire LLM Chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load the Groq And OpenAI Api Key
os.environ['OPEN_AI_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Objectbox VectorstoreDB with Llama3 Demo")

#LLM MODEL
llm=ChatGroq(groq_api_key = groq_api_key,
             model_name = "Llama3-8b-8192")


#Chat Prompt Template, generic prompt for any context
prompt = ChatPromptTemplate.from_template(

    """
    Answer the questions based on the provided contenxt only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """#

#for context, two things are required 
)

## Vector Embedding and Object Vectorstore db

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()  #first embedding is OPENAI embedding
        #data ingestion
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") #loader will be loading all PDF from forlder us_cesnus
        st.session_state.docs = st.session_state.loader.load() ## Documents Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size  = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents,  st.session_state.embeddings, embeddings_dimensions = 768)

input_prompt = st.text_input("Enter Your Question From Documents")
if st.button("Documents Embedding"): #when a button is pressed, it should load all doc and perfrom vector embedding
    vector_embedding()
    st.write("ObjectBox Database is ready")

import time  

if input_prompt:
    documents_chain = create_stuff_documents_chain(llm, prompt) #pass LLM and gen prompt
    retriever = st.session_state.vectors.as_retriever() #as_retriever func is used to whatever vector database is, and we want to retrieve data/info it will create an interface on top of vector db 
    retrieval_chain = create_retrieval_chain(retriever, documents_chain) #retrieval chain consists of db retriever chain and documents chain
    start = time.process_time()

    response = retrieval_chain.invoke({'input':input_prompt})

    print("Response time :",  time.process_time()-start) 
    st.write(response['answer'])

    #With a streamlit expander
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")

