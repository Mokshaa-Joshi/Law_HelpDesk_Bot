import streamlit as st
import pymongo
import pinecone
import openai
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["Saudi_Arabia_Law"]
collection = db["pdf_chunks"]
pdf_collection = db["pdf_repository"]  

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-qna"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  
        metric="cosine"
    )

index = pc.Index(index_name)

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def chunks_exist(pdf_name):
    return collection.count_documents({"pdf_name": pdf_name}) > 0

def insert_chunks(chunks, pdf_name):
    if not chunks_exist(pdf_name):
        for chunk in chunks:
            collection.insert_one({"pdf_name": pdf_name, "text": chunk})

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = openai_client.embeddings.create(input=[chunk], model="text-embedding-ada-002").data[0].embedding
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

def list_stored_pdfs():
    return pdf_collection.distinct("pdf_name")

def store_pdf(pdf_name, pdf_data):
    if pdf_collection.count_documents({"pdf_name": pdf_name}) == 0:
        pdf_collection.insert_one({"pdf_name": pdf_name, "pdf_data": pdf_data})

def query_vectors(query, selected_pdf):
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

def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

st.markdown(
    "<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        store_pdf(uploaded_file.name, uploaded_file.read())

        if not chunks_exist(uploaded_file.name):
            chunks = process_pdf(temp_pdf_path)
            insert_chunks(chunks, uploaded_file.name)
            store_vectors(chunks, uploaded_file.name)
            st.success("PDF uploaded and processed!")
        else:
            st.info("This PDF has already been processed!")

        selected_pdf = uploaded_file.name

elif pdf_source == "Choose from the Document Storage":
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available in the repository. Please upload one.")

input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

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

if st.button("Get Answer"):
    if selected_pdf and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        
        response = query_vectors(detected_lang, selected_pdf)

        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
