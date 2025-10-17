"""Chunk the raw JSON files"""
import os
import json
from loguru import logger
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import tiktoken

load_dotenv()


def tiktoken_len(text):
    # Using cl100k_base encoding which is a good general-purpose tokenizer
    # This works well for estimating token counts even with Ollama models
    tokens = tiktoken.get_encoding("cl100k_base").encode(
        text,
    )
    return len(tokens)

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
                    'lr_no': doc.get('lr_no', '').replace("Release No.", ""),
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
        length_function=tiktoken_len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_documents = text_splitter.split_documents(documents)

    logger.info(f"Created {len(split_documents)} chunks from {len(documents)} documents")

    return split_documents

