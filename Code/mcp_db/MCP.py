from typing import List, Dict, Any
import os
from mcp.server.fastmcp import FastMCP
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import tool
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_qdbe2_Lx5TuKQbDTeqr5iUUd2rrrkzthbaGbj7LtnKZ7FtWa5QmsxTgnpBbjzhHBhc5Xe")
INDEX_NAME = "mcpvecdb"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print("Connected to Pinecone and index initialized")

mcp = FastMCP("pinecone-mcp")


embedder = OllamaEmbeddings(model="llama3")

@mcp.tool()
async def search_docs(query: str) -> str:
    """Search the indexed documents for relevant information based on a query"""
    query_embedding = embedder.embed_query(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    if not results.get("matches"):
        return "No relevant documents found."

    top_texts = [match["metadata"]["text"] for match in results["matches"]]
    combined_response = "\n\n---\n\n".join(top_texts)
    return f"Top results:\n\n{combined_response}"


@mcp.tool()
async def describe_index_stats() -> Dict[str, Any]:
    """Provides statistics about the data in the index"""
    stats = index.describe_index_stats()
    return {
        "total_records": stats.get("total_vector_count", 0),
        "namespaces": list(stats.get("namespaces", {}).keys())
    }


@mcp.tool()
async def search_records(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Searches for records in an index based on a text query"""
    results = index.query(vector=[0] * 4096, top_k=top_k, include_metadata=True, query=query)
    return [{"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})} for m in results["matches"]]


@mcp.tool()
async def upload_document(file_path: str) -> str:
    """Uploads a PDF document and adds its content to the Pinecone index using text chunks"""
    
    loader = PyPDFLoader("roadmap.pdf")
    documents = loader.load()

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    records = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        print(f"Embedding chunk {i+1}/{len(chunks)}")
        
        vector = embedder.embed_query(text)  # Use embed_query from OllamaEmbeddings
        record = {
            "id": str(uuid4()),
            "values": vector,
            "metadata": {"text": text}
        }
        records.append(record)

    if records:
        index.upsert(vectors=records)
        return f"Uploaded document with {len(records)} chunks."
    return "No content found in document."


if __name__ == "__main__":
    async def test():
        result = await search_docs("wwhat is week 15 about ?")
        print(result)
    asyncio.run(test())
