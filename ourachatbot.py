import os
import glob
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load JSON files handling multiple JSON objects per file
files = glob.glob("*.json")
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

print(f"Loaded {len(documents)} documents")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# Build the chatbot
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

# Chat loop
print("Oura Health Chatbot ready! Type 'quit' to exit.")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = chain.invoke(question)
    print(f"Bot: {answer.content}\n")