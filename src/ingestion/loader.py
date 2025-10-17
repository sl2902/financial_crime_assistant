"""Load chunked documents into Qdrant"""
import os
import json
import glob
from loguru import logger
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.ingestion.chunker import(
    recursive_chunking,
)

load_dotenv()


class QdrantStoreManager:
    """Class for Vector storage"""

    def __init__(
            self,
            collection_name: str = "financial_crimes",
            location: str = ":memory:",
            path: str = None,
            url: str = None,
            dim: int = 1536, 
            distance: Distance = Distance.COSINE, 
            embedding_model: str = "text-embedding-3-small"
            ):
        self.collection_name = collection_name
        self.location = location
        self.path = path
        self.url = url
        self.dim = dim
        self.distance = distance
        self.embedding_model = embedding_model
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(location=location)
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
    
    def create_collection(self):
        """Create Qdrant collection"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=self.distance),
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection might already exist: {e}")
    
    def get_vector_store(self):
        """Embed the chunks"""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

def load_to_qdrant(chunks: List[Document], vector_store_manager: QdrantVectorStore):
    """Load chunks into Qdrant"""
    vector_store = vector_store_manager.get_vector_store()

    vector_store.add_documents(chunks)
    
    logger.info(f"Loaded {len(chunks)} chunks to Qdrant collection: {vector_store_manager.collection_name}")

def vector_store_retriever(vector_store_manager: QdrantVectorStore, search_kwargs: Dict[str, int] = {"k": 3}):
    """Define the Qdrant Vector Store retriever"""
    vector_store = vector_store_manager.get_vector_store()

    return vector_store.as_retriever(search_kwargs=search_kwargs)


if __name__ == "__main__":

    vector_store_manager = QdrantStoreManager(path="./qdrant_data")

    vector_store_manager.create_collection()

    batch_files = glob.glob("data/raw/sec_releases_batch_*.json")
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            data = json.load(f)
            releases = data["releases"]
        chunks = recursive_chunking(releases, chunk_size=750, chunk_overlap=100)

        load_to_qdrant(chunks, vector_store_manager)

    retriever = vector_store_retriever(vector_store_manager)
    answer = retriever.invoke("What kind of crimes have people been accused of?")

    logger.info(answer)

    logger.success("Ingestion complete!")