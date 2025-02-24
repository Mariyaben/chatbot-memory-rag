import os
import pandas as pd
import chromadb
import zipfile
from sentence_transformers import SentenceTransformer

def extract_zip(zip_path, extract_to="extracted_data"):
    """Extracts all CSV files from a ZIP archive to a specified folder."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted ZIP contents to {extract_to}")
    return extract_to

def load_data(folder_path):
    """Load all CSV files from a folder and combine them into a dictionary."""
    data_dict = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            data_dict[file] = df
    return data_dict

def preprocess_text(df):
    """Flatten dataframe text columns into a list of text entries."""
    text_data = []
    for col in df.select_dtypes(include=["object"]):
        text_data.extend(df[col].dropna().astype(str).tolist())
    return text_data

def store_embeddings_in_chroma(data_dict, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings and store in ChromaDB."""
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("esg_data")
    memory_collection = client.get_or_create_collection("esg_chat_memory")  

    model = SentenceTransformer(model_name)
    
    for filename, df in data_dict.items():
        text_data = preprocess_text(df)
        embeddings = model.encode(text_data).tolist()
        
        for idx, emb in enumerate(embeddings):
            collection.add(
                ids=[f"{filename}_{idx}"],
                embeddings=[emb],
                metadatas=[{"filename": filename, "text": text_data[idx]}]
            )
    
    print("Embeddings stored successfully in ChromaDB!")

zip_file_path = "esg.zip"  
extract_folder = extract_zip(zip_file_path)
esg_data = load_data(extract_folder)
store_embeddings_in_chroma(esg_data)
