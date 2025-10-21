"""Financial Crime Compliance Assistant - Streamlit Interface with RAG Mode Selection"""
import streamlit as st
import pandas as pd
from pathlib import Path
import markdown2
import time

from src.ingestion.loader import QdrantStoreManager, vector_store_retriever as get_retriever
from src.rag.retriever import FinancialCrimeRAGSystem

def custom_success(html_content: str, title="Success!"):
    """Custom success box that works with HTML content"""
    st.markdown(
        f"""
        <div style="
            background-color: #d4edda;
            border-color: #c3e6cb;
            border: 1px solid transparent;
            border-radius: 0.25rem;
            padding: 0.75rem 1.25rem;
            margin-bottom: 1rem;
            color: #155724;
        ">
            <strong>✅ {title}</strong>
            <hr style="margin: 10px 0; border-color: #c3e6cb;">
            {html_content}
        """,
        unsafe_allow_html=True
    )

# Page config
st.set_page_config(
    page_title="Financial Crime Assistant",
    page_icon="🏦",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    with st.spinner("Loading system..."):
        store_manager = QdrantStoreManager(path="./qdrant_data")
        retriever = get_retriever(store_manager, search_kwargs={"k": 5})
        st.session_state.rag_system = FinancialCrimeRAGSystem(retriever=retriever)
        st.session_state.messages = []

# Sidebar - Evaluation Metrics and Settings
st.sidebar.title("⚙️ Settings")

# RAG Mode Selector
st.sidebar.subheader("🤖 RAG Mode")
rag_mode = st.sidebar.radio(
    "Select RAG Mode:",
    ["Plain RAG", "Agentic RAG"],
    help="""
    **Plain RAG**: Uses only the document retrieval system (faster, focused on SEC documents)
    
    **Agentic RAG**: Can search web for current information and use multiple tools (slower, more comprehensive)
    """
)

# Display which mode is active
if rag_mode == "Plain RAG":
    st.sidebar.info("📚 Using document retrieval only")
else:
    st.sidebar.success("🌐 Using agent with web search + documents")

st.sidebar.divider()

# Evaluation Results Section
st.sidebar.title("📊 Evaluation Results")

# Load evaluation results
try:
    eval_df = pd.read_csv("evaluation/retriever/retriever_evaluation_results.csv")
    
    st.sidebar.subheader("Baseline vs Advanced")
    
    # Create comparison
    metrics = ["faithfulness", "answer_relevancy", "context_entity_recall", "context_recall", "noise_sensitivity"]
    # comparison_data = {
    #     "Metric": metrics,
    #     "Baseline": [eval_df[m].mean() 
    #                  for m in metrics 
    #                  if m in eval_df.columns and eval_df.filter(like='retriever', axis=1)['retriever'].iloc[0] == 'naive_retriever'
    #                 ],
    #     "Advanced": [eval_df[m].mean() 
    #                  for m in metrics 
    #                  if m in eval_df.columns and eval_df.filter(like='retriever', axis=1)['retriever'].iloc[1] != 'naive_retriever'
    #                 ]
    # }
    
    comparison_df = eval_df.T[1:][eval_df.T[1:].index.isin(metrics)].rename(columns={0:"Baseline", 1: "Advanced"})
    comparison_df = comparison_df.reset_index().rename(columns={"index": "Metric"})
    comparison_df["Baseline"] = comparison_df["Baseline"].astype(float)
    comparison_df["Advanced"] = comparison_df["Advanced"].astype(float)
    if any(comparison_df["Baseline"]) and any(comparison_df["Advanced"]):
        comparison_df["Improvement"] = ((comparison_df["Advanced"] - comparison_df["Baseline"]) / comparison_df["Baseline"] * 100).round(1)
        
        st.sidebar.dataframe(
            comparison_df.style.format({
                "Baseline": "{:.3f}",
                "Advanced": "{:.3f}",
                "Improvement": "{:+.1f}%"
            }),
            hide_index=True
        )
        
        # Bar chart
        st.sidebar.subheader("Performance Comparison")
        chart_df = comparison_df.set_index("Metric")[["Baseline", "Advanced"]]
        st.sidebar.bar_chart(chart_df, stack=False)

        # Cost vs Latency performance
        metrics = ["langsmith_total_cost_usd", "langsmith_total_latency"]
        comparison_df = eval_df.T[1:][eval_df.T[1:].index.isin(metrics)].rename(columns={0:"Baseline", 1: "Advanced"})
        comparison_df = comparison_df.reset_index().rename(columns={"index": "Metric"})
        comparison_df["Baseline"] = comparison_df["Baseline"].astype(float)
        comparison_df["Advanced"] = comparison_df["Advanced"].astype(float)
        if any(comparison_df["Baseline"]) and any(comparison_df["Advanced"]):
            comparison_df["Improvement"] = ((comparison_df["Advanced"] - comparison_df["Baseline"]) / comparison_df["Baseline"] * 100).round(1)
            
            st.sidebar.dataframe(
                comparison_df.style.format({
                    "Baseline": "{:.3f}",
                    "Advanced": "{:.3f}",
                    "Improvement": "{:+.1f}%"
                }),
                hide_index=True
            )
        
        # Bar chart
        st.sidebar.subheader("Cost vs Latency Comparison")
        chart_df = comparison_df.set_index("Metric")[["Baseline", "Advanced"]]
        st.sidebar.bar_chart(chart_df, stack=False)


except FileNotFoundError:
    st.sidebar.warning("Evaluation results not found. Run evaluation first.")

# Main interface
st.title("🏦 Financial Crime Compliance Assistant")

# Show current mode in main area
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("Ask questions about SEC enforcement cases, financial crimes, and compliance regulations.")
with col2:
    mode_badge = "🤖 Agentic" if rag_mode == "Agentic RAG" else "📚 Plain"
    st.info(f"Mode: {mode_badge}")

# Sample questions
with st.expander("📋 Sample Questions"):
    if rag_mode == "Plain RAG":
        st.markdown("""
        **Best for Plain RAG (document-based):**
        - What is securities fraud and what are typical penalties?
        - Tell me about recent insider trading cases
        - What was the outcome of case LR-26415?
        - How do Ponzi scheme penalties compare to securities fraud?
        - What are common patterns in insider trading cases?
        """)
    else:
        st.markdown("""
        **Best for Agentic RAG (web + documents):**
        - What financial crime news happened this week? (uses web search)
        - What are recent SEC enforcement actions? (uses web search)
        - Compare historical penalties with current trends
        - What penalties were issued for insider trading in 2025? (searches documents first)
        - Latest developments in cryptocurrency fraud cases (uses web search)
        """)

# Query interface
st.subheader("💬 Ask a Question")

# Text input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What are typical penalties for insider trading?"
)

# Query button with mode indicator
execution_time = None
button_text = f"Submit ({rag_mode})"
if st.button(button_text, type="primary") or (query and len(st.session_state.messages) == 0):
    if query:
        with st.spinner(f"{'Searching documents...' if rag_mode == 'Plain RAG' else 'Searching documents and web...'}"):
            # Call appropriate method based on mode
            try:
                start = time.time()
                if rag_mode == "Plain RAG":
                    # Use plain RAG (document retrieval only)
                    response = st.session_state.rag_system.query(query)
                else:
                    # Use agentic RAG (can use web search + documents)
                    response = st.session_state.rag_system.agent_query(query)
            
                
                if hasattr(response, "content"):
                    response = response.content
                
                # Store in session with mode info
                st.session_state.messages.append({
                    "question": query,
                    "answer": response,
                    "mode": rag_mode
                })
                end = time.time()  
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try switching RAG modes or rephrasing your question.")
        execution_time = end - start

# Display conversation history
if st.session_state.messages:
    st.subheader("📝 Response")
    
    # Show latest response
    latest = st.session_state.messages[-1]
    
    # Show which mode was used
    mode_used = latest.get("mode", "Unknown")
    st.markdown(f"**Question:** (answered using {mode_used})")
    st.info(latest["question"])
    
    st.markdown("**Answer:**")
    answer_text = latest["answer"]

    if isinstance(answer_text, dict):
        # Agentic RAG response
        display_text = answer_text.get("result", str(answer_text))
        tools_used = answer_text.get("tools_used", [])
        sources = answer_text.get("sources", [])
        rag_query = answer_text.get("rag_query", "")
    else:
        # Plain RAG response (string)
        display_text = str(answer_text)
        tools_used = []
        sources = []
        rag_query = ""
    html = markdown2.markdown(display_text, extras=['tables', 'fenced-code-blocks'])
    custom_success(html)

    result = latest["answer"]
    if isinstance(result, dict) and result.get("tools_used") and execution_time:
        st.caption(f"🔧 Tools: {', '.join(result['tools_used'])} 🕒 Execution Time: {round(execution_time, 2)} seconds")
        st.write(f'Number of sources {len(result.get("sources"))}')
    else:
        if execution_time:
            st.caption(f"🕒 Execution Time: {round(execution_time, 2)} seconds")
    # Show sources (expandable)
    if isinstance(result, dict) and result.get("sources"):
        with st.expander("📚 View Sources"):
            for i, doc in enumerate(result["sources"], 1):
                lr_no = doc.metadata.get('lr_no', 'Unknown')
                url = doc.metadata.get('url', 'N/A')
                date = doc.metadata.get('date', 'N/A')
                crime = ', '.join(doc.metadata.get('crime_type', []))
                
                st.markdown(f"**{i}. {lr_no}** ({date})")
                st.markdown(f"Crime Type: {crime}")
                if rag_query:
                    st.markdown(f"Query: {rag_query}")
                st.markdown(f"[View on SEC.gov]({url})")
                st.markdown(doc.page_content[:200] + "...")
                st.divider()
    
    # Show conversation history
    if len(st.session_state.messages) > 1:
        with st.expander("📜 Previous Questions"):
            for i, msg in enumerate(reversed(st.session_state.messages[:-1])):
                msg_mode = msg.get("mode", "Unknown")
                st.markdown(f"**Q{len(st.session_state.messages)-i-1}** [{msg_mode}]: {msg['question']}")
                
                # Show preview
                preview = str(msg['answer'])[:200] + "..." if len(str(msg['answer'])) > 200 else str(msg['answer'])
                st.text(preview)
                st.divider()

# Clear button
if st.session_state.messages:
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()

# Footer
st.divider()
st.caption(f"Financial Crime Compliance Assistant | Mode: {rag_mode} | Powered by RAG + LangGraph")