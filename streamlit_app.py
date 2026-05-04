import os
import glob
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Oura Ring Health Chatbot")
st.write("Ask me anything about your health data!")

def convert_numbers(obj):
    if isinstance(obj, dict):
        return {k: convert_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numbers(i) for i in obj]
    elif isinstance(obj, str):
        try:
            return int(obj)
        except ValueError:
            try:
                return float(obj)
            except ValueError:
                return obj
    return obj

@st.cache_resource(ttl=0)
def load_chain():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base_dir, "*.json"))
    documents = []
    all_data = []
    for file in files:
        with open(file, "r") as f:
            content = f.read()
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            try:
                obj, idx = decoder.raw_decode(content, pos)
                obj = convert_numbers(obj)
                documents.append(Document(page_content=json.dumps(obj), metadata={"source": file}))
                all_data.append(obj)
                pos += idx
            except json.JSONDecodeError:
                pos += 1

    # Pre-compute key stats
    sleep_records = [(o["day1"], o["score"]) for o in all_data if "score" in o and "day1" in o and "stress_high" not in o and "steps" not in o]
    readiness_records = [(o["day1"], o["score"]) for o in all_data if "score" in o and "day1" in o and "avg_readiness_score" not in o and "steps" not in o and "stress_high" not in o and "avg_sleep_score" not in o]
    activity_records = [(o["day1"], o["steps"]) for o in all_data if "steps" in o and "day1" in o]
    stress_records = [(o["day1"], o["stress_high"]) for o in all_data if "stress_high" in o and "day1" in o]

    stats_lines = []
    if sleep_records:
        min_s = min(sleep_records, key=lambda x: x[1])
        max_s = max(sleep_records, key=lambda x: x[1])
        stats_lines.append(f"Lowest sleep score: {min_s[1]} on {min_s[0][:10]}")
        stats_lines.append(f"Highest sleep score: {max_s[1]} on {max_s[0][:10]}")
    if activity_records:
        max_steps = max(activity_records, key=lambda x: x[1])
        stats_lines.append(f"Most steps in a day: {max_steps[1]} on {max_steps[0][:10]}")
    if stress_records:
        max_stress = max(stress_records, key=lambda x: x[1])
        min_stress = min(stress_records, key=lambda x: x[1])
        stats_lines.append(f"Highest stress: {max_stress[1]} on {max_stress[0][:10]}")
        stats_lines.append(f"Lowest stress: {min_stress[1]} on {min_stress[0][:10]}")

    precomputed_stats = "\n".join(stats_lines)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_template("""
You are a health data analyst. Here are pre-computed key stats from the full dataset:
{stats}

Answer the question based on the stats above and the following additional context:
{context}

Question: {question}

Answer:
""")

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "stats": RunnableLambda(lambda _: precomputed_stats)
        }
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