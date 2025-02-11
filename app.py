import streamlit as st
import pymongo
import pinecone
import openai
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  

# Load environment variables (for local testing)
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["Saudi_Arabia_Law"]
collection = db["pdf_chunks"]
pdf_collection = db["pdf_repository"]  

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
        dimension=1536,  
        metric="cosine"
    )

index = pc.Index(index_name)

# OpenAI API Key
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to process PDF and chunk it
def process_pdf(pdf_path, chunk_size=500):
    """Extracts text from a PDF and splits it into chunks."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to check if chunks already exist
def chunks_exist(pdf_name):
    """Checks if chunks for a given PDF already exist in MongoDB."""
    return collection.count_documents({"pdf_name": pdf_name}) > 0

# Function to store chunks in MongoDB (only if not already stored)
def insert_chunks(chunks, pdf_name):
    """Inserts extracted chunks into MongoDB."""
    if not chunks_exist(pdf_name):
        for chunk in chunks:
            collection.insert_one({"pdf_name": pdf_name, "text": chunk})

# Function to convert text into vectors and store in Pinecone
def store_vectors(chunks, pdf_name):
    """Embeds text chunks using OpenAI and stores them in Pinecone."""
    for i, chunk in enumerate(chunks):
        vector = openai_client.embeddings.create(input=[chunk], model="text-embedding-ada-002").data[0].embedding
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Function to retrieve stored PDFs from MongoDB repository
def list_stored_pdfs():
    """Fetches unique stored PDF names from the database."""
    return pdf_collection.distinct("pdf_name")

# Function to store uploaded PDFs in MongoDB repository
def store_pdf(pdf_name, pdf_data):
    """Saves uploaded PDFs to MongoDB to avoid re-uploading the same file."""
    if pdf_collection.count_documents({"pdf_name": pdf_name}) == 0:
        pdf_collection.insert_one({"pdf_name": pdf_name, "pdf_data": pdf_data})

# Function to query Pinecone and get relevant answers
def query_vectors(query, selected_pdf):
    """Retrieves relevant document chunks from Pinecone and uses OpenAI for answers."""
    vector = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002").data[0].embedding
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]

        combined_text = "\n\n".join(matched_texts)

        prompt = (
            f"Based on the following legal document ({selected_pdf}), provide an accurate and well-reasoned answer:\n\n"
            f"{combined_text}\n\n"
            f"User's Question: {query}"
        )

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in legal analysis."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        return response
    else:
        return "No relevant information found in the selected document."

# Function to translate text
def translate_text(text, target_language):
    """Translates text to the specified language (English or Arabic)."""
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# ==============================
# 🚀 Streamlit UI
# ==============================
st.title("AI-Powered Saudi Arabia Law HelpDesk")

# Sidebar - List all stored PDFs
st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None

# Select PDF source
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

# 📌 Upload a new PDF
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

        # Assign uploaded file name to selected_pdf
        selected_pdf = uploaded_file.name

# 📌 Browse previously uploaded PDFs
elif pdf_source == "Choose from the Document Storage":
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available in the repository. Please upload one.")

# 🟢 Input Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)

# 📌 Adjust input box for Arabic
if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    query_html = """
    <style>
    .stTextInput>div>div>input {
        direction: rtl;
        text-align: right;
    }
    </style>
    """
    st.markdown(query_html, unsafe_allow_html=True)
else:
    query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

# 📌 Get Answer Button
if st.button("Get Answer"):
    if selected_pdf and query:
        # Convert Arabic input to English for better AI processing
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        
        # Process the query
        response = query_vectors(detected_lang, selected_pdf)

        # Translate response back to Arabic if Arabic was selected
        if input_lang == "Arabic":
            response = translate_text(response, "ar")
            
            # Display response in RTL format
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
