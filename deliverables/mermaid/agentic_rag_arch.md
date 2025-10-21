flowchart TD
    Start([User Query]) --> ParseQuery[Parse Query<br/>Extract LR Numbers<br/>Detect Temporal Indicators]
    
    ParseQuery --> Agent{Agent<br/>GPT-4o-mini<br/>Tool Selection}
    
    Agent -->|LR Number Found| FilterPath[Create Metadata Filter<br/>lr_no = LR-XXXXX]
    Agent -->|Historical/Case Query| RAGPath[Use RAG Tool]
    Agent -->|Current/Recent Query| TavilyPath[Use Tavily Tool]
    
    FilterPath --> FilterSearch[Filter-Only Search<br/>No Embedding Needed]
    RAGPath --> SemanticSearch[Embed Query<br/>Semantic Search<br/>k=10]
    
    FilterSearch --> QdrantDB[(Qdrant<br/>Vector Store<br/>550 chunks)]
    SemanticSearch --> QdrantDB
    
    QdrantDB --> RAGDocs[Retrieved SEC Documents<br/>with Metadata]
    
    TavilyPath --> TavilyAPI[Tavily Web Search<br/>max_results=5]
    TavilyAPI --> WebResults[Web Search Results<br/>Current News]
    
    RAGDocs --> Generate[Generate Response<br/>GPT-4o-mini<br/>+ Add Citations]
    WebResults --> Generate
    
    Generate --> FormatResponse[Format Output<br/>Fix $ signs<br/>Add Source URLs]
    
    FormatResponse --> Response([Final Answer<br/>+ Sources<br/>+ Tools Used])
    
    Response --> User([User])
    
    style Agent fill:#e1f5ff
    style FilterSearch fill:#ffe1cc
    style SemanticSearch fill:#ffe1cc
    style QdrantDB fill:#f0f0f0
    style Generate fill:#e1ffe1
    style Response fill:#fff3e1
    style User fill:#d4edda