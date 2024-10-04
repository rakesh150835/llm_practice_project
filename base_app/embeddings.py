import os
import torch
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber
import numpy as np
import faiss


# Load the SentenceTransformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    except Exception as e:
        print(f"Error in extract_text_from_pdf func: {e}")


def save_embeddings(embeddings_list, filenames):
    try:
        # Create 'vectorstore' directory if it doesn't exist
        if not os.path.exists('vectorstore'):
            os.makedirs('vectorstore')

        for i, embeddings in enumerate(embeddings_list):
            np.save(f'vectorstore/embeddings_{i}.npy', embeddings)
        
        # Save filenames
        np.save('vectorstore/filenames.npy', filenames)

    except Exception as e:
        print(f"Error in save_embeddings func: {e}")


# Function to create embeddings for multiple PDFs
def create_embeddings_for_pdfs(directory):
    try:
        embeddings_list = []
        filenames = []
        # Initialize the RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
        )
        
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory, filename)
                text = extract_text_from_pdf(pdf_path)  # PDF extraction function
                # Use the recursive text splitter to split the text
                chunks = text_splitter.split_text(text)
                
                # Create embeddings for each chunk
                embeddings = model.encode(chunks)
                embeddings_list.append(embeddings)
                filenames.append(filename)
        
        save_embeddings(embeddings_list, filenames)
        create_faiss_index(embeddings_list)

    except Exception as e:
        print(f"Error in create_embeddings_for_pdfs func: {e}")



def save_faiss_index(index, index_file):
    try:
        # Create 'vectorstore' directory if it doesn't exist
        if not os.path.exists('vectorstore'):
            os.makedirs('vectorstore')
        faiss.write_index(index, index_file)

    except Exception as e:
        print(f"Error in save_faiss_index func: {e}")


# Create a FAISS index for efficient similarity search
def create_faiss_index(embeddings_list):
    try:
        dimension = embeddings_list[0].shape[1]  # Embedding dimension
        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        for embeddings in embeddings_list:
            index.add(embeddings)  

        save_faiss_index(index, 'vectorstore/faiss_index.bin')
    
    except Exception as e:
        print(f"Error in create_faiss_index func: {e}")





