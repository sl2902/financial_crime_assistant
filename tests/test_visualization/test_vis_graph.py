# In Python console or script
import os
from src.rag.retriever import FinancialCrimeRAGSystem
from src.visualization.graph_viz import GraphVisualizer
from src.ingestion.loader import QdrantStoreManager, vector_store_retriever

store_manager = QdrantStoreManager(path="./qdrant_data")

retriever = vector_store_retriever(store_manager)
rag_system = FinancialCrimeRAGSystem(retriever=retriever)

answer_data = rag_system.agent_query("Who worked with Ponzi defendants?")

print("Tools used:", answer_data['tools_used'])
graph_results = answer_data['graph_results']

print(f"Graph results type: {type(graph_results)}")
print(f"Graph results length: {len(graph_results) if graph_results else 0}")

if graph_results:
    print("First result:", graph_results["results"][0])
    
    viz = GraphVisualizer()
    html_path = viz.visualize(graph_results["results"], "test_graph.html")
    
    print(f"HTML created: {html_path}")
    print(f"File size: {os.path.getsize(html_path)} bytes")
    
    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(html_path)}")