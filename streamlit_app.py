"""Financial Crime Compliance Assistant - Streamlit Interface with RAG Mode Selection"""
import streamlit as st
import pandas as pd
from pathlib import Path

from src.ingestion.loader import QdrantStoreManager, vector_store_retriever as get_retriever
from src.rag.retriever import FinancialCrimeRAGSystem
from src.utils.document_utils import fix_merged_text

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
    metrics = ["faithfulness", "answer_relevancy", "context_entity_recall", "context_recall"]
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
button_text = f"Submit ({rag_mode})"
if st.button(button_text, type="primary") or (query and len(st.session_state.messages) == 0):
    if query:
        with st.spinner(f"{'Searching documents...' if rag_mode == 'Plain RAG' else 'Searching documents and web...'}"):
            # Call appropriate method based on mode
            try:
                if rag_mode == "Plain RAG":
                    # Use plain RAG (document retrieval only)
                    response = st.session_state.rag_system.query(query)
                else:
                    # Use agentic RAG (can use web search + documents)
                    response = st.session_state.rag_system.agent_query(query)
                
                # Apply text fixing
                response = fix_merged_text(response)
                
                if hasattr(response, "content"):
                    response = response.content
                
                # Store in session with mode info
                st.session_state.messages.append({
                    "question": query,
                    "answer": response,
                    "mode": rag_mode
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try switching RAG modes or rephrasing your question.")

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
    
    # Try to display the answer
    answer_text = str(latest["answer"])
    
    # Use code block for now to avoid formatting issues
    st.code(answer_text, language='markdown')
    
    # Option to view in different formats
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 View as Text"):
            st.text(answer_text)
    with col2:
        if st.button("📄 View as Markdown"):
            try:
                st.markdown(answer_text)
            except:
                st.error("Cannot render as markdown")
    with col3:
        if st.button("📋 Copy to Clipboard"):
            st.code(answer_text, language=None)
            st.success("Ready to copy!")
    
    # Debug section
    with st.expander("🔍 Debug Information", expanded=False):
        st.write(f"**Mode Used:** {mode_used}")
        st.write(f"**Response Length:** {len(answer_text)} characters")
        
        # Check for common issues
        issues = []
        if "alongwith" in answer_text.lower():
            issues.append("Merged words detected")
        if any(ord(c) > 127 for c in answer_text[:500]):
            issues.append("Unicode characters detected")
        
        if issues:
            st.warning(f"Issues found: {', '.join(issues)}")
        else:
            st.success("No formatting issues detected")
    
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