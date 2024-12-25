# main.py
import os
import numpy as np
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from pdf_reader import load_pdf  # Importing PDF reader utility

# Load environment variables (e.g., OpenAI API Key)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Function to generate embeddings for a list of text chunks
def generate_embeddings(text_chunks):
    """Generate embeddings for text using OpenAI Embeddings."""
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_documents(text_chunks)

# Function to create and return a FAISS index from embeddings
def create_faiss_index(embeddings_list):
    """Create a FAISS index for efficient vector search."""
    embedding_matrix = np.array(embeddings_list).astype("float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index

# Function to perform a search query against the FAISS index
def search_query(query, index, embeddings, k=5):
    """Search the FAISS index for the most relevant results."""
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return indices, distances

# Conversational search system
def conversational_search():
    """Run the conversational search."""
    pdf_text = load_pdf("example.pdf")  # Load your PDF file
    text_chunks = split_text(pdf_text)  # Split text into chunks

    embeddings_list = generate_embeddings(text_chunks)  # Generate embeddings for text chunks
    index = create_faiss_index(embeddings_list)  # Create FAISS index for search

    print("Welcome to the document search engine! Type 'exit' to quit.")
    while True:
        query = input("Enter your query: ")
        
        if query.lower() in ['exit', 'quit']:
            break
        
        indices, distances = search_query(query, index, OpenAIEmbeddings())
        top_chunks = [text_chunks[i] for i in indices[0]]
        
        print("Top search results:\n")
        for chunk in top_chunks:
            print(chunk[:300])  # Display the first 300 characters of the top chunk

if __name__ == "__main__":
    conversational_search()
