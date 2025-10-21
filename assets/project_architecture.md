flowchart TB
 subgraph Data_Collection["1. Data Collection"]
        Scraper["sec_scraper.py<br>Async Scraping<br>500 documents"]
        SEC["SEC.gov<br>Litigation Releases"]
        RawData[("Local storage<br>data/raw/<br>sec_releases_batch_*.json")]
  end
 subgraph Preprocessing["2. Data Preprocessing"]
        Preprocessor["preprocesser.py<br>Unicode Cleaning<br>Entity Extraction<br>Crime Categorization"]
        ProcessedData[("Local storage<br>data/processed/<br>sec_releases_batch_*_clean.json")]
  end
 subgraph Chunking["3. Chunking & Embedding"]
        Chunker["chunker.py<br>RecursiveCharacterTextSplitter<br>750 tokens/100 overlap"]
        Embedder["OpenAI Embeddings<br>text-embedding-3-small<br>1536 dimensions"]
        Loader["loader.py<br>QdrantStoreManager"]
  end
 subgraph Vector_Store["4. Vector Store"]
        Qdrant[("Qdrant<br>Local Storage<br>~550 chunks<br>financial_crimes collection")]
  end
 subgraph RAG_System["5. RAG System - Dual Mode"]
        RAG["RAG Pipeline<br>Plain RAG or Agentic RAG<br>Vector Search + Reranking<br>GPT-4o-mini Generation"]
        Query(["User Query"])
        WebSearch["Tavily Web Search<br>Current Events"]
        Response(["Response<br>+ Sources<br>+ Metadata"])
  end
 subgraph Evaluation["6. Evaluation & Monitoring"]
        LangSmith["LangSmith<br>Tracing &amp; Monitoring"]
        RAGAS["RAGAS Evaluation<br>Faithfulness: 0.925<br>Answer Relevancy: 0.820<br>Context Recall: 0.733"]
  end
    SEC -- Web Scraping --> Scraper
    Scraper -- JSON --> RawData
    RawData -- Load --> Preprocessor
    Preprocessor -- Cleaned JSON --> ProcessedData
    ProcessedData -- Load --> Chunker
    Chunker -- Chunks --> Embedder
    Embedder -- Vectors --> Loader
    Loader -- Persist --> Qdrant
    Query --> RAG
    RAG --> Qdrant & Response
    RAG -. Agentic Mode .-> WebSearch
    Response -.-> LangSmith & RAGAS
    Response --> Display["Streamlit UI<br>Display Results + Metrics"]



