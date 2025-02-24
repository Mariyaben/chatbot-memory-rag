import os
import sys
import pandas as pd
import pysqlite3  
sys.modules["sqlite3"] = pysqlite3  

import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  
if not openai_api_key:
    openai_api_key = st.secrets["OPENAI_API_KEY"]  


def preprocess_text(df):
    return [text for col in df.select_dtypes(include=["object"]) 
            for text in df[col].dropna().astype(str).tolist()]

def store_embeddings_in_chroma(data_dict, model_name="all-MiniLM-L6-v2"):
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("esg_data")
    model = SentenceTransformer(model_name)
    
    for filename, df in data_dict.items():
        text_data = preprocess_text(df)
        if not text_data:
            continue
        
        embeddings = model.encode(text_data).tolist()
        for idx, emb in enumerate(embeddings):
            collection.add(
                ids=[f"{filename}_{idx}"],
                embeddings=[emb],
                metadatas=[{"filename": filename, "text": text_data[idx]}]
            )

st.set_page_config(page_title="ESG Chatbot", page_icon="🌍", layout="wide")
st.title("🌍 ESG Chatbot - AI-powered Insights")
st.write("**Ask me anything about ESG data or upload CSV files!**")

uploaded_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
data_dict = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        data_dict[uploaded_file.name] = df
    store_embeddings_in_chroma(data_dict)
    st.success("Data successfully uploaded and processed!")

##MEMORY-AWARE RETRIEVER

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("esg_data")
memory_collection = chroma_client.get_or_create_collection("memory_rag")

def retrieve_documents(query, k=5):
    results = collection.query(query_texts=[query], n_results=k)
    if results.get("metadatas") and len(results["metadatas"]) > 0:
        return [doc["text"] for doc in results["metadatas"][0] if "text" in doc]
    else:
        return ["No relevant data found."]


def update_memory(query, response):
    memory_data = memory_collection.get()
    memory_ids = memory_data["ids"] if memory_data and "ids" in memory_data else []
    
    memory_collection.add(
        ids=[f"memory_{len(memory_ids)}"],
        embeddings=[[0] * 384],  
        metadatas=[{"query": query, "response": response}]
    )

def retrieve_memory(query):
    results = memory_collection.query(query_texts=[query], n_results=3)
    return [doc["response"] for doc in results["metadatas"][0]] if results.get("metadatas") else []

###AI QUERY PROCESSING 

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)

prompt_template = PromptTemplate(
    template="""
    You are an ESG data expert. Answer the user's query based on the retrieved documents.

    Query: {query}
    Memory Context:
    {memory_context}

    Retrieved Context:
    {retrieved_context}

    Provide a clear and concise response.
    """,
    input_variables=["query", "memory_context", "retrieved_context"]
)

def answer_query(query):
    memory_context = "\n".join(retrieve_memory(query))
    retrieved_context = "\n".join(retrieve_documents(query))
    
    final_prompt = prompt_template.format(
        query=query,
        memory_context=memory_context,
        retrieved_context=retrieved_context
    )
    
    response = llm.invoke([SystemMessage(content=final_prompt)]).content
    update_memory(query, response)
    return response



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question:")
if query:
    with st.spinner("Fetching answer..."):
        response = answer_query(query)
    
    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("ESG Bot", response))

for role, msg in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"**🧑‍💼 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")

st.write("💬 **Chat session will remember previous interactions!**")
