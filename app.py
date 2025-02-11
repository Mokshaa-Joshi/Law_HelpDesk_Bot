import streamlit as st
import pymongo
import pinecone
import openai
import PyPDF2
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["pdf_database"]
collection = db["pdf_chunks"]

# Pinecone Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("pdf-qna")

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to process PDF and chunk it
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to store chunks in MongoDB
def insert_chunks(chunks):
    for chunk in chunks:
        collection.insert_one({"text": chunk})

# Function to convert text into vectors and store in Pinecone
def store_vectors(chunks):
    for i, chunk in enumerate(chunks):
        vector = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")["data"][0]["embedding"]
        index.upsert([(f"doc-{i}", vector, {"text": chunk})])

# Function to query Pinecone and get answers
def query_vectors(query):
    vector = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    results = index.query(vector=vector, top_k=1, include_metadata=True)
    return results["matches"][0]["metadata"]["text"] if results["matches"] else "No relevant information found."

# Function to list PDF files from a GitHub repository
def list_pdf_files():
    GITHUB_REPO = "https://api.github.com/repos/YOUR_USERNAME/YOUR_REPO/contents/pdfs"
    response = requests.get(GITHUB_REPO)
    if response.status_code == 200:
        return [file["name"] for file in response.json() if file["name"].endswith(".pdf")]
    return []

# Function to download PDF from GitHub
def download_pdf(filename):
    file_url = f"https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/pdfs/{filename}"
    response = requests.get(file_url)
    file_path = f"temp_{filename}"
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path

# Streamlit UI
st.title("AI-Powered Q&A Chatbot 📚")

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Repository"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())

        chunks = process_pdf(f"temp_{uploaded_file.name}")
        insert_chunks(chunks)
        store_vectors(chunks)
        st.success("PDF uploaded and processed!")

elif pdf_source == "Choose from Repository":
    pdf_list = list_pdf_files()
    selected_pdf = st.selectbox("Select a PDF", pdf_list)

    if st.button("Download and Process"):
        file_path = download_pdf(selected_pdf)
        chunks = process_pdf(file_path)
        insert_chunks(chunks)
        store_vectors(chunks)
        st.success("PDF processed successfully!")

query = st.text_input("Ask a question:")
if st.button("Get Answer"):
    response = query_vectors(query)
    st.write(f"**Answer:** {response}")
