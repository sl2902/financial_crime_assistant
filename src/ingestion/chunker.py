"""Chunk the raw JSON files"""
import os
from loguru import logger
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


class QdrantVectoreStore:
    """Class for Vector storage"""

    def __init__(
            self,
            collection_name: str = "financial_crimes",
            location: str = ":memory:",
            url: str = None,
            dim: int = 1536, 
            distance: Distance = Distance.COSINE, 
            embedding_model: str = "text-embedding-3-small"
            ):
        self.location = location
        self.url = url
        self.dim = dim
        self.distance = distance
        self.embedding_model = embedding_model
        if url:
            self.client = QdrantClient(url=url)
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

def vector_store_retriever(vector_store_manager: QdrantVectoreStore, search_kwargs: Dict[str, int] = {"k": 3}):
    """Define the Qdrant Vector Store retriever"""
    vector_store = vector_store_manager.get_vector_store()

    return vector_store.as_retriever(search_kwargs=search_kwargs)

def recursive_chunking(
        docs: List[Dict], 
        chunk_size: int = 750, 
        chunk_overlap: int = 100
    ) -> List[Document]:
    """Recursively chunk the document"""
    # Convert List[Dict] to LangChain Document
    documents = []
    for doc in docs:
        if 'content' not in doc or not doc['content']:
            logger.warning(f"Skipping document without content: {doc.get('lr_no', 'unknown')}")
            continue
        if 'success' in doc and not doc['success']:
            logger.warning(f"Skipping document as there is no content")
            continue
        documents.append(
            Document(
                page_content=doc['content'],
                metadata={
                    'lr_no': doc.get('lr_no', ''),
                    'date': doc.get('date', ''),
                    'url': doc.get('main_link', ''),
                    'title': doc.get('title', ''),
                    'see_also': doc.get('see_also', ''),
                    'source': 'SEC'
                }
            )
        )
    
    logger.info(f"Converting {len(documents)} documents to chunks")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_documents = text_splitter.split_documents(documents)

    logger.info(f"Created {len(split_documents)} chunks from {len(documents)} documents")

    return split_documents
