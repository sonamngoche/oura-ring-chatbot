import os
import glob
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Oura Ring Health Chatbot")
st.write("Ask me anything about your health data!")

@st.cache_resource
def load_chain():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base_dir, "*.json"))
    documents = []
    for file in files:
        with open(file, "r") as f:
            content = f.read()
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            try:
                obj, idx = decoder.raw_decode(content, pos)
                documents.append(Document(page_content=json.dumps(obj), metadata={"source": file}))
                pos += idx
            except json.JSONDecodeError:
                pos += 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following health data context:
    {context}

    Question: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your health data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})