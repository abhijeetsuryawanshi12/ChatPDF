import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

##Setup Streamlit
st.title("ChatPDF")
st.write("Upload Pdf's and chat with their content")

#Input the Groq API key
api_key = st.text_input("Enter your Groq API key", type="password")
if api_key:
    llm=ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    session_id=st.text_input("Enter your session id", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files=st.file_uploader("Upload PDF's",type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as f:
                f.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    #Split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embeddings=embeddings)
        retriever = vectorstore.as_retriever()
    
    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed and otherwise return it as it is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    history_aware_retriever=create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    ##Answer question prompt

    system_prompt= (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you"
        "don't know. UUse three sentences maximum and keep the answer concise"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever, chain)




