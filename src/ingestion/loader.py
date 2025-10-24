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
from qdrant_client.models import (
    Filter, 
    FieldCondition, 
    MatchValue, 
    Match,
    PayloadSchemaType
)

from src.ingestion.chunker import(
    recursive_chunking,
)

load_dotenv()

qdrant_filepath = "qdrant_data"

class QdrantStoreManager:
    """Class for Vector storage"""

    def __init__(
            self,
            collection_name: str = "financial_crimes",
            location: str = ":memory:",
            path: str = None,
            url: str = None,
            api_key: str = None,
            dim: int = 1536, 
            distance: Distance = Distance.COSINE, 
            embedding_model: str = "text-embedding-3-small"
            ):
        self.collection_name = collection_name
        self.location = location
        self.path = path
        self.url = url
        self.api_key = api_key
        self.dim = dim
        self.distance = distance
        self.embedding_model = embedding_model
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"Connected to Qdrant Cloud: {url}")
        elif path:
            self.client = QdrantClient(path=path)
            logger.info(f"Using local Qdrant storage: {path}")
        else:
            self.client = QdrantClient(location=location)
            logger.info(f"Using local system memory")
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
    
    def collection_exists(self) -> bool:
        """Check if collection already exists"""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except:
            return False
    
    def create_collection(self):
        """Create Qdrant collection"""
        if self.collection_exists():
            logger.info(f"Collection '{self.collection_name}' already exists")
            return
        
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=self.distance),
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def get_vector_store(self):
        """Embed the chunks"""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
    
    def search_by_lr_number(self, lr_no: str) -> Tuple[List, int|None]:
        """Search Qdrant store using LR No"""
        flt = Filter(should=[FieldCondition(key="metadata.lr_no", match=MatchValue(value=lr_no))])
        results = self.client.scroll(
            collection_name="financial_crimes",
            scroll_filter=flt,
            limit=5,
            with_payload=True
        )
        return results

def load_to_qdrant(chunks: List[Document], vector_store_manager: QdrantVectorStore):
    """Load chunks into Qdrant"""
    vector_store = vector_store_manager.get_vector_store()

    vector_store.add_documents(chunks)
    
    logger.info(f"Loaded {len(chunks)} chunks to Qdrant collection: {vector_store_manager.collection_name}")

def vector_store_retriever(vector_store_manager: QdrantVectorStore, search_kwargs: Dict[str, int] = {"k": 3}):
    """Define the Qdrant Vector Store retriever"""
    vector_store = vector_store_manager.get_vector_store()

    return vector_store.as_retriever(search_kwargs=search_kwargs)

def test_loader(collection_name: str, query: str) -> None:
    """Test loader"""
    processed_filepath = "data/processed/"
    vector_store_manager = QdrantStoreManager(path=qdrant_filepath)
    if vector_store_manager.client.collection_exists(collection_name):
        retriever = vector_store_retriever(vector_store_manager)
        docs = retriever.invoke(query)
        if docs:
            # Check first retrieved chunk
            print("=== CHUNK CONTENT ===")
            print(docs[0].page_content)
            print("\n=== METADATA ===")
            print(docs[0].metadata)

def main():

    processed_filepath = "data/processed/"
    collection_name = "financial_crimes"
    vector_store_manager = QdrantStoreManager(path=qdrant_filepath)


    if vector_store_manager.collection_exists():
        logger.warning("Collection already exists. Delete ./qdrant_data to re-ingest.")
        try:
            vector_store_manager.client.delete_collection("financial_crimes")
            logger.warning("Deleted old collection")
        except:
            pass


    vector_store_manager.create_collection()

    logger.info("Loading SEC releases into Qdrant...")
    batch_files = glob.glob(f"{processed_filepath}/sec_releases_batch_*.json")
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            data = json.load(f)
            releases = data["releases"]
        logger.info(f"Chunking documents for batch {batch_file}...")
        chunks = recursive_chunking(releases, chunk_size=750, chunk_overlap=100)

        logger.info("Loading to Qdrant...")
        load_to_qdrant(chunks, vector_store_manager)
    # all_docs = []
    # for file in glob.glob(f"{processed_filepath}/*_clean.json"):
    #     logger.info(f"Loading {file}")
    #     with open(file) as f:
    #         data = json.load(f)
    #         all_docs.extend(data["releases"])

    # logger.info(f"Total docs: {len(all_docs)}")

    # # Chunk
    # chunks = recursive_chunking(all_docs)
    # logger.info(f"Total chunks: {len(chunks)}")


    logger.success(f"Ingestion complete! Data persisted to {qdrant_filepath}")
    
    retriever = vector_store_retriever(vector_store_manager)

    # # user test loading
    # answer = retriever.invoke("What kind of crimes have people been accused of?")

    # logger.info(answer)

    case = 'LR-26115'
    results = vector_store_manager.search_by_lr_number(case)
    if results[0]:
        logger.info(f"{case} is in Qdrant")
        logger.info(f"Title: {results[0][0].payload.get('metadata').get('title')}")
    else:
        logger.warning(f"{case} NOT in Qdrant - needs to be ingested!")


if __name__ == "__main__":
    main()
    # test_loader("financial_crimes", "Robert Allen Stanford penalties")