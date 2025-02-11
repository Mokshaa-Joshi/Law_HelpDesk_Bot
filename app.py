import streamlit as st
import pymongo
import pinecone
import openai
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  # For Arabic-English translation

# Load environment variables (for local testing)
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["Saudi_Arabia_Law"]
collection = db["pdf_chunks"]
pdf_collection = db["pdf_repository"]  # Collection for storing PDFs

# Pinecone Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-qna"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust if using a different model
        metric="cosine"
    )

index = pc.Index(index_name)

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to process PDF and chunk it
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to check if chunks already exist
def chunks_exist(pdf_name):
    return collection.count_documents({"pdf_name": pdf_name}) > 0

# Function to store chunks in MongoDB (only if not already stored)
def insert_chunks(chunks, pdf_name):
    if not chunks_exist(pdf_name):
        for chunk in chunks:
            collection.insert_one({"pdf_name": pdf_name, "text": chunk})

# Function to convert text into vectors and store in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = openai.embeddings.create(input=[chunk], model="text-embedding-ada-002").data[0].embedding
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Function to retrieve stored PDFs from MongoDB repository
def list_stored_pdfs():
    return pdf_collection.distinct("pdf_name")  # Fetch unique PDF names

# Function to store uploaded PDFs in MongoDB repository
def store_pdf(pdf_name, pdf_data):
    if pdf_collection.count_documents({"pdf_name": pdf_name}) == 0:
        pdf_collection.insert_one({"pdf_name": pdf_name, "pdf_data": pdf_data})

# Function to query Pinecone and get accurate answers
def query_vectors(query, selected_pdf):
    vector = openai.embeddings.create(input=[query], model="text-embedding-ada-002").data[0].embedding
    results = index.query(vector=vector, top_k=3, include_metadata=True, filter={"pdf_name": selected_pdf})
    
    if results["matches"]:
        best_match = results["matches"][0]["metadata"]["text"]
        prompt = f"Based on the following extracted text from {selected_pdf}, answer the query accurately:\n\n{best_match}\n\nQuery: {query}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an AI assistant that provides highly accurate answers from legal documents."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    else:
        return "No relevant information found."

# Function to translate text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.title("AI-Powered Saudi Arabia Law HelpDesk")

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Repository"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Store PDF in MongoDB repository
        store_pdf(uploaded_file.name, uploaded_file.read())

        if not chunks_exist(uploaded_file.name):
            chunks = process_pdf(temp_pdf_path)
            insert_chunks(chunks, uploaded_file.name)
            store_vectors(chunks, uploaded_file.name)
            st.success("PDF uploaded and processed!")
        else:
            st.info("This PDF has already been processed!")

elif pdf_source == "Choose from Repository":
    pdf_list = list_stored_pdfs()
    selected_pdf = st.selectbox("Select a PDF", pdf_list)

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if selected_pdf and query:
        lang_option = st.radio("Choose Response Language", ["English", "Arabic"])
        response = query_vectors(query, selected_pdf)
        
        if lang_option == "Arabic":
            response = translate_text(response, "ar")

        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
