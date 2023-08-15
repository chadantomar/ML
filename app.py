import streamlit as st
from PyPDF2 import PdfReader
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from htmltemplate import css, bot_template, user_template


def main():
    st.set_page_config(page_title="Chat with you PDF", page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    st.header("Chat with your PDF :books:")
    user_question=st.text_input("Ask your Question")
    if user_question:
        handle_Question(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    
    st.write(bot_template.replace("{{MSG}}","hello human"),unsafe_allow_html=True)

    st.write(user_template.replace("{{MSG}}","Hello robot"),unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Documents")
        files= st.file_uploader("Upload Pdfs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing.."):
                data= get_text_pdf(files)
                text_as_chunk = get_text_chunk(data)
                
                vectorstore= get_vectorstore(text_as_chunk)
                st.session_state.conversation = get_conversationchain(vectorstore)


def get_text_pdf(files):
    text=""
    for file in files:
        pdf_reader= PdfReader(file)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunk(text):
    text_spliter= CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunk_text= text_spliter.split_text(text)
    return chunk_text

key= ""

def get_vectorstore(chunktext_arrays):
    embeddings= OpenAIEmbeddings(openai_api_key=key)
    vectorstore= FAISS.from_texts(texts=chunktext_arrays, embedding=embeddings)
    return vectorstore

def get_conversationchain(vectorstore):
    llm=ChatOpenAI()
    memory= ConversationBufferMemory(memory_key="chat_history", return_message=True)
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory= memory
    )
    return conversation_chain

def handle_Question(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.write(response)
if __name__== "__main__":
    main()
