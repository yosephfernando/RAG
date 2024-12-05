from dotenv import load_dotenv
import os
import arxiv
from pathlib import Path
import re

from pinecone import Pinecone, ServerlessSpec

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser

load_dotenv()

# Instantiate `PDFReader` from LlamaHub
loader = PDFReader()

# Load HNSW PDF from LFS
documents = loader.load_data(file=Path('./1603.09320v4.pdf'))

def clean_up_text(content: str) -> str:
    """
    Remove unwanted characters and patterns in text input.

    :param content: Text input.
    
    :return: Cleaned version of original text input.
    """

    # Fix hyphenated words broken by newline
    content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)

    # Remove specific unwanted patterns and characters
    unwanted_patterns = [
        "\\n", "  —", "——————————", "—————————", "—————",
        r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7'
    ]
    for pattern in unwanted_patterns:
        content = re.sub(pattern, "", content)

    # Fix improperly spaced hyphenated words and normalize whitespace
    content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
    content = re.sub(r'\s+', ' ', content)

    return content

# Call function
cleaned_docs = []
for d in documents: 
    cleaned_text = clean_up_text(d.text)
    d.text = cleaned_text
    cleaned_docs.append(d)

# Initialize Pinecone client
PC_API_KEY = os.getenv("PINECONE_API_KEY")
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
pc_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pc_index)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define the initial pipeline
pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95, 
            embed_model=embed_model,
        ),
        embed_model,
    ],
    vector_store=vector_store
)

pipeline.run(documents=cleaned_docs)
pc_index.describe_index_stats()
