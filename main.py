from dotenv import load_dotenv
import os

from pinecone import Pinecone, ServerlessSpec

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

import ollama

load_dotenv()

PC_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PC_API_KEY)
index_name = "llama-integration-example"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Set embed_model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Initialize Ollama client
ollama_model = "llama3.2:latest"

# Initialize Pinecone vector store
pc_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pc_index)

# Function to retrieve top-k relevant documents from Pinecone
def retrieve_documents(query, top_k=5):
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

    results = retriever.retrieve(query)
    return results


# Function to use Ollama for generative QA based on retrieved documents
def rag_inference(query):
    # Step 1: Retrieve relevant documents from Pinecone
    documents = retrieve_documents(query)
    
    # Step 2: Extract text from each document and combine them
    context = "\n".join([doc.text for doc in documents])  # Assuming 'text' is the attribute holding the content
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: Use Ollama to generate an answer
    response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
    
    generated_answer = response['message']['content']
    return generated_answer

# Example usage
query = "What title of the document ?"
answer = rag_inference(query)
print(answer)