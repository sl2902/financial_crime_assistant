"""Langchain Document utils"""
import os
import glob
from typing import Any, List, Dict, Optional

from langchain.schema import Document

batch_files = glob.glob("data/raw/sec_releases_batch_*.json")

def json_to_documents(json_data: List[Dict]) -> List[Document]:
    """Convert raw JSON to full LangChain Documents"""
    documents = []
    for doc in json_data:
        if 'content' not in doc or not doc['content']:
            continue

        if 'success' in doc and not doc['success']:
            continue
        
        documents.append(
            Document(
                page_content=doc['content'],
                metadata={
                    'lr_no': doc.get('lr_no', '').replace("Release No.", ""),
                    'date': doc.get('date', ''),
                    'url': doc.get('url', ''),
                    'title': doc.get('title', ''),
                    'see_also': doc.get('see_also', ''),
                    'source': 'SEC'
                }
            )
        )
    return documents

def json_to_document(document: Dict) -> Optional[Document]:
    """Convert a single JSON object to Document"""
    if 'content' in document and 'success' in document and document['success']:
        return Document(
                    page_content=document['content'],
                    metadata={
                        'lr_no': document.get('lr_no', '').replace("Release No.", ""),
                        'date': document.get('date', ''),
                        'url': document.get('url', ''),
                        'title': document.get('title', ''),
                        'see_also': document.get('see_also', ''),
                        'source': 'SEC'
                    }
                )
    return None