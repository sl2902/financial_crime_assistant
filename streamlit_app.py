"""Financial Crime Compliance Assistant - Streamlit Interface with RAG Mode Selection"""
import os
import streamlit as st
import pandas as pd
from pathlib import Path
import markdown2
import time
from datetime import datetime
import atexit
from loguru import logger
from pydantic import ValidationError

from qdrant_client.models import (
    Filter, 
    FieldCondition, 
    MatchValue, 
    MatchAny, 
    Range,
    DatetimeRange
)

from src.ingestion.loader import QdrantStoreManager, vector_store_retriever as get_retriever
from src.rag.retriever import FinancialCrimeRAGSystem
from src.schemas.rag_schemas import QueryInput
from src.visualization.graph_viz import GraphVisualizer

@st.cache_data
def display_graph_visualization(graph_results, answer_data):
    """Display interactive graph visualization in Streamlit.
    
    Args:
        graph_results: Results from graph query tool
        answer_data: Full answer data including tools used
    """
    # Check if graph tool was used
    # logger.info(f'Here inside display_graph_visualization {graph_results}')
    tools_used = answer_data.get('tools_used', [])
    
    if 'search_knowledge_graph' not in tools_used:
        return  # Don't show graph if tool wasn't used
    
    # Check if we have results
    if not graph_results or len(graph_results) == 0:
        st.info("🕸️ No network connections found for this query")
        return
    
    # Create graph visualization
    st.subheader("🕸️ Network Connections")
    
    with st.expander("📊 View Interactive Graph", expanded=True):
        try:
            # Initialize visualizer
            viz = GraphVisualizer(height="650px", width="100%")
            
            # Create graph
            html_file = viz.visualize(graph_results)
            
            # if os.path.exists(html_file):
            #     # Read HTML
            #     with open(html_file, 'r') as f:
            #         html_content = f.read()

            # Display in Streamlit
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            
            st.components.v1.html(html_content, height=800, scrolling=True)
            
            # Add legend
            st.markdown("""
            **Legend:**
            - 👤 **Red nodes** = People
            - 🏢 **Teal nodes** = Companies
            - ⚖️ **Yellow nodes** = Cases
            - 💰 **Mint nodes** = Penalties
            
            **Interactions:**
            - 🖱️ Drag nodes to rearrange
            - 🔍 Scroll to zoom
            - 💡 Hover for details
            """)

            st.download_button(
                "📥 Download Graph HTML (if not visible above)",
                data=html_content,
                file_name="network_graph.html",
                mime="text/html"
            )
            
        except Exception as e:
            st.error(f"Error creating graph visualization: {e}")


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

def build_filter_from_ui():
    """Build Qdrant filter from UI selections"""
    conditions = []
    
    # Crime type filter
    if crime_types:
        conditions.append(
            FieldCondition(
                key="metadata.crime_type",
                match=MatchAny(any=crime_types)
            )
        )

    # Dat filter
    if start_date or end_date:
        date_range = {}
        if start_date:
            date_range["gte"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            date_range["lte"] = end_date.strftime("%Y-%m-%d")
        
        conditions.append(
            FieldCondition(
                key="metadata.date",
                range=DatetimeRange(
                    gte=date_range["gte"],
                    lte=date_range["lte"],
                    lt=None,
                    gt=None,
                ),
            )
        )

    # if start_year or end_year:
    #     years = []
    #     if start_year and end_year:
    #         years = list(range(start_year, end_year + 1))
    #     elif start_year:
    #         years = [start_year]
    #     elif end_year:
    #         years = [end_year]
        
    #     conditions.append(
    #         FieldCondition(
    #             key="metadata.date",
    #             match=MatchAny(any=[str(year) for year in years])
    #         )
    #     )

    # Amount filter
    if penalty_category != "All":
        conditions.append(
            FieldCondition(
                key="metadata.penalty_category",
                match=MatchValue(value=penalty_category)
            )
        )
    # if amount_min > 0 or amount_max > 0:
    #     amount_range = {}
    #     if amount_min > 0:
    #         amount_range["gte"] = amount_min
    #     if amount_max > 0:
    #         amount_range["lte"] = amount_max
        
    #     conditions.append(
    #         FieldCondition(
    #             key="metadata.amounts",
    #             range=Range(**amount_range)
    #         )
    #     )

    # logger.info(f'Filter conditions {conditions}')
    if conditions:
        return Filter(must=conditions)
    return None

# Page config
st.set_page_config(
    page_title="Financial Crime Assistant",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def get_qdrant_client():
    return QdrantStoreManager(path="./qdrant_data")

# Initialize session state
if 'rag_system' not in st.session_state:
    with st.spinner("Loading system..."):
        store_manager = get_qdrant_client()
        retriever = get_retriever(store_manager, search_kwargs={"k": 3})
        st.session_state.rag_system = FinancialCrimeRAGSystem(retriever=retriever)
        st.session_state.messages = []

@atexit.register
def close_qdrant():
    try:
        store_manager.client.close()
    except Exception:
        pass

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

st.sidebar.subheader("🔍 Advanced Filters")

# Crime Type filter
crime_types = st.sidebar.multiselect(
    "Crime Type:",
    options=[
        "Ponzi Scheme",
        "Insider Trading", 
        "Securities Fraud",
        "Market Manipulation",
        "Fraud (General)",
        "Cryptocurrency Fraud"
    ],
    key="crime_types",
    help="Filter by type of financial crime"
)

# # Date range filter
# col1, col2 = st.sidebar.columns(2)
# with col1:
#     start_year = st.sidebar.selectbox(
#         "From Year:",
#         options=[None, 2020, 2021, 2022, 2023, 2024, 2025],
#         index=0
#     )
# with col2:
#     end_year = st.sidebar.selectbox(
#         "To Year:",
#         options=[None, 2020, 2021, 2022, 2023, 2024, 2025],
#         index=0
#     )
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "From:",
        value=None,
        min_value=datetime(2023, 1, 1),
        max_value=datetime(2025, 12, 31),
        key="start_date"
    )
with col2:
    end_date = st.date_input(
        "To:",
        value=None,
        min_value=start_date,
        max_value=datetime(2025, 12, 31),
        key="end_date"
    )

# Amount range filter
penalty_category = st.sidebar.selectbox(
    "Penalty Range:",
    options=["All", "Under $100K", "$100K - $1M", "$1M - $10M", "Over $10M"],
    key="penalty_type"
)
# st.sidebar.subheader("💰 Penalty Amount")
# amount_min = st.sidebar.number_input(
#     "Min Amount ($):",
#     min_value=0,
#     value=0,
#     step=10000,
#     key="amount_min"
# )
# amount_max = st.sidebar.number_input(
#     "Max Amount ($):",
#     min_value=0,
#     value=0,
#     step=10000,
#     help="0 = no limit",
#     key="amount_max"
# )

# Clear filters button
if st.sidebar.button("🗑️ Clear Filters"):
    for key in ["crime_types", "start_date", "end_date", "amount_min", "amount_max", "penalty_type"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Evaluation Results Section
st.sidebar.title("📊 Evaluation Results")

# Load evaluation results
try:
    eval_df = pd.read_csv("evaluation/retriever/retriever_evaluation_results.csv")
    
    st.sidebar.subheader("Baseline vs Advanced")
    
    # Create comparison
    metrics = ["faithfulness", "factual_correctness", "answer_relevancy", "context_entity_recall", "context_recall", "noise_sensitivity"]
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
        chart_df["Advanced"] = chart_df["Advanced"] / chart_df["Baseline"]
        chart_df["Baseline"] = chart_df["Baseline"] / chart_df["Baseline"]
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

# avoid the button from being 'clicked' more than once
if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = None

# Query button with mode indicator
execution_time = None
button_text = f"Submit ({rag_mode})"
if st.button(button_text, type="primary") or (query and len(st.session_state.messages) == 0):
    if query  and st.session_state.last_processed_query != query:
        st.session_state.last_processed_query = query
        with st.spinner(f"{'Searching documents...' if rag_mode == 'Plain RAG' else 'Searching documents and web...'}"):
            # Call appropriate method based on mode
            try:
                start = time.time()
                query = QueryInput(
                    question=query,
                    limit = 3
                )
                if rag_mode == "Plain RAG":
                    # Use plain RAG (document retrieval only)
                    response = st.session_state.rag_system.query(query.question, filter=build_filter_from_ui(), limit=query.limit)
                else:
                    # Use agentic RAG (can use web search + documents)
                    response = st.session_state.rag_system.agent_query(query.question, filter=build_filter_from_ui())
            
                
                if hasattr(response, "content"):
                    response = response.content
                
                # Store in session with mode info
                st.session_state.messages.append({
                    "question": query,
                    "answer": response,
                    "mode": rag_mode
                })
                end = time.time()
                execution_time = end - start
            except ValidationError as err:
                logger.error(f"Citation validation failed: {err}")
                st.error(" An error occurred — see details below")
                import traceback
                st.code(traceback.format_exc())
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
    st.info(latest["question"].question)
    
    st.markdown("**Answer:**")
    answer_text = latest["answer"]

    if isinstance(answer_text, dict):
        # Agentic RAG response
        display_text = answer_text.get("result", str(answer_text))
        tools_used = answer_text.get("tools_used", [])
        sources = answer_text.get("sources", [])
        rag_query = answer_text.get("rag_query", "")
        graph_results = answer_text.get("graph_results", {})
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
        # st.write(f'Number of sources {len(result.get("sources"))}')
    else:
        if execution_time:
            st.caption(f"🕒 Execution Time: {round(execution_time, 2)} seconds")
    
    logger.info(f"Tool used {answer_text.get('tools_used')}")
    if answer_text.get("tools_used") and 'search_knowledge_graph' in answer_text.get("tools_used"):
        display_graph_visualization(answer_text.get("graph_results", {}).get("results", []), answer_text)
    
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