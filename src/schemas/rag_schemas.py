"""Pydantic schemas for RAG system responses"""
from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    """Citation for a source document"""
    lr_number: str = Field(description="Case number like LR-26123")
    url: str = Field(description="Full SEC.gov URL")  # ← Changed from HttpUrl
    title: Optional[str] = Field(default=None, description="Case title")

class StructuredAnswer(BaseModel):
    """Structured answer with inline citations"""
    answer_text: str = Field(
        description="""Answer formatted in markdown with:
- Section headers (## Historical Penalties, ## Current Trends, ## Comparison)
- Bullet points for individual cases (use * or -)
- Each case as a separate bullet with sub-bullets for details
- Inline citations after each fact: [LR-XXXXX](url) or [SourceName](url)

Example format:
## Historical Penalties
* **Barry Siegel (2024)** [LR-26123](url):
  - Charges: Securities violations
  - Penalty: $112,868.95 civil fine
  
## Current Trends
* **Recent enforcement surge** [Fenergo](url):
  - 417% increase in penalties
  - $1.23 billion total"""
    )
    sources_used: List[Citation] = Field(
        description="List of all sources cited in the answer"
    )

class RAGResponse(BaseModel):
    """Complete RAG system response"""
    result: str = Field(description="Final formatted answer")
    tools_used: List[str] = Field(description="Tools used by agent")
    sources: List[Citation] = Field(description="Source documents")
    rag_query: str = Field(description="Query used for retrieval")