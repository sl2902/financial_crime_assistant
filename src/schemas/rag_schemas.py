"""Pydantic schemas for RAG system responses"""
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List, Optional
from datetime import datetime
import re

class Citation(BaseModel):
    """Citation for a source document"""
    lr_number: str = Field(description="Case number like LR-26123")
    url: str = Field(description="Full SEC.gov URL")
    title: Optional[str] = Field(default=None, description="Case title")

class StructuredAnswer(BaseModel):
    """Structured answer with inline citations"""
    answer_text: str = Field(
        description="""Answer formatted in markdown with:
- Section headers (## Historical Penalties, ## Current Trends, ## Comparison)
- Bullet points for individual cases (use * or -)
- Each case as a separate bullet with sub-bullets for details
- Inline citations after each fact: [LR-XXXXX](url) or [SourceName](https://www.sec.gov/enforcement-litigation/litigation-releases/LR-XXXXX)
- For external web sources: ONLY cite if from reputable sources (Reuters, Bloomberg, WSJ, etc.)

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

    @field_validator('answer_text')
    def validate_citations(cls, v: str, info: ValidationInfo):
        """Check that all citations in text match sources_used"""
        sources = info.data.get('sources_used') if info.data else None
        if not sources:
            return v
        
        # Extract all LR citations from text
        lr_citations = set(re.findall(r'LR-\d+', v))
        
        # Get LR numbers from sources
        valid_lrs = {src.lr_number  for src in sources}
        
        # Check for hallucinated citations
        invalid = lr_citations - valid_lrs
        if invalid:
            raise ValueError(f"Answer contains citations not in sources: {invalid}")
        
        return v

class RAGResponse(BaseModel):
    """Complete RAG system response"""
    result: str = Field(description="Final formatted answer")
    tools_used: List[str] = Field(description="Tools used by agent")
    sources: List[Citation] = Field(description="Source documents")
    rag_query: str = Field(description="Query used for retrieval")

class QueryInput(BaseModel):
    """Input schema for RAG queries"""
    question: str = Field(
        min_length=3,
        max_length=500,
        description="User's question"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )
    crime_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by crime types"
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Start date for filtering"
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End date for filtering"
    )
    penalty_category: Optional[str] = Field(
        default=None,
        description="Penalty amount category"
    )
    
    @field_validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()
    
    # @field_validator('end_date')
    # def end_after_start(cls, v, values):
    #     if v and 'start_date' in values and values['start_date']:
    #         if v < values['start_date']:
    #             raise ValueError('End date must be after start date')
    #     return v
    
    # @field_validator('crime_types')
    # def valid_crime_types(cls, v):
    #     if v:
    #         valid_types = [
    #             "Ponzi Scheme",
    #             "Insider Trading",
    #             "Securities Fraud",
    #             "Market Manipulation",
    #             "Fraud (General)",
    #             "Cryptocurrency Fraud"
    #         ]
    #         for crime_type in v:
    #             if crime_type not in valid_types:
    #                 raise ValueError(f"Invalid crime type: {crime_type}")
    #     return v