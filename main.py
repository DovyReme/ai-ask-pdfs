# import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import openai
import chromadb
import docx2txt
from chromadb import Client
from chromadb.config import Configuration
from chromadb.client import Client
from PyMuPDF import fitz  # for reading PDF files
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize ChromaDB client
client = chromadb.Client(Configuration(endpoint="http://localhost:8000"))

def extract_text_from_pdf(file_path):
    text = ""
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def insert_document_to_chroma_db(document_id, content, api_key):
    # Create a document entry in ChromaDB
    entry = {
        'document_id': document_id,
        'content': content
    }
    response = client.post_document(api_key, entry)
    if response.status_code != 200:
        raise ValueError(f"Failed to create document entry: {response.text}")

def main():
    data_folder = "data"
    api_key = openai_api_key  # or groq_api_key, depending on your usage
    
    for root, _, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            try:
                content = process_file(file_path)
                document_id = os.path.basename(file_path)
                insert_document_to_chroma_db(document_id, content, api_key)
                print(f"Successfully inserted document: {document_id}")
            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")

if __name__ == "__main__":
    main()