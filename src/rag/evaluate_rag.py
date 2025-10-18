"""Evaluate RAG using golden testset"""
import os
import json
import pandas as pd
from uuid import uuid4
from operator import itemgetter
from loguru import logger
import time
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv

from langsmith import Client, tracing_context
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_cohere import CohereRerank
from langchain.storage import InMemoryStore

from ragas import EvaluationDataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy, 
    ContextEntityRecall, 
    NoiseSensitivity
)
from ragas import evaluate, RunConfig

from src.ingestion.loader import QdrantStoreManager

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Create unique session ID for this evaluation run
EVALUATION_SESSION_ID = uuid4().hex[:8]
os.environ["LANGCHAIN_PROJECT"] = f"Financial-Crimes-Advanced-Retrieval-Eval-{EVALUATION_SESSION_ID}"

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

class EvaluateRAGASDataset:
    def __init__(self):
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, seed=42)
    
    def get_rag_prompt(self):
        if self.rag_prompt:
            return self.rag_prompt
    
    def get_chat_model(self):
        if self.llm:
            return self.llm

    def make_lcel_chain(
            self,
            rag_prompt: ChatPromptTemplate, 
            chat_model: ChatOpenAI,
            retriever: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make a LCEL chain
        """
        return (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
        )

    def cast_text(self, response: AIMessage) -> str:
        """Cast a response to a string"""
        return response.content if isinstance(response, AIMessage) else response
    
    
    def naive_retriever_chain(
        # rag_prompt: ChatPromptTemplate, 
        # chat_model: ChatOpenAI,
        self, 
        vector_store: QdrantVectorStore,
        k: int = 10
    ) -> Dict[str, Any]: 
        """Create a naive retriever"""
        naive_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        return naive_retriever
    
    def contextual_compression_retriever_chain(
            self,
            rag_prompt: ChatPromptTemplate, 
            chat_model: ChatOpenAI, 
            naive_retriever: Dict[str, Any],
            k: int = 5,
    ) -> Dict[str, Any]:
        """Create a contextual compression retriever"""
        compressor = CohereRerank(model="rerank-v3.5")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=naive_retriever,
            search_kwargs={"k": k}
        )

        return compression_retriever


    def create_evaluation_dataset(
            self,
            dataset: Dict, 
            retriever: Dict[str, Any], 
            rag_prompt: ChatPromptTemplate, 
            chat_model: ChatOpenAI,
            retriever_name: str = None,
            session_id: str = None,
    ) -> EvaluationDataset:
        """Create a evaluation dataset with session tracking"""
        logger.info(f"Make LCEL RAG chain for `{retriever_name}`")
        lcel_chain = self.make_lcel_chain(rag_prompt, chat_model, retriever)
        # dataset["response"], dataset["retrieved_contexts"] = None, None
        
        # Process each test case with metadata tracking
        for doc in dataset:
            user_input = doc.get("user_input")
            if user_input:
                # Invoke with metadata in config (correct way for LangSmith)
                retrieved_docs = lcel_chain.invoke(
                    {"question": user_input},
                    config={
                        "metadata": {
                            "retriever": retriever_name,
                            "session_id": session_id or EVALUATION_SESSION_ID,
                            "task": "retrieval_evaluation"
                        },
                        "tags": [retriever_name, f"session-{session_id or EVALUATION_SESSION_ID}"]
                    }
                )
                doc['response'] = self.cast_text(retrieved_docs["response"])
                doc['retrieved_contexts'] = [
                    context.page_content for context in retrieved_docs["context"]
                ]
                # doc.eval_sample.response = self.cast_text(retrieved_docs["response"])
                # doc.eval_sample.retrieved_contexts = [
                #     context.page_content for context in retrieved_docs["context"]
                # ]
        # dataset['reference_contexts'] = dataset['reference_contexts'].apply(
        #                                         lambda x: json.loads(x) if isinstance(x, str) else x
        # )
        evaluation_dataset = EvaluationDataset.from_pandas(pd.DataFrame(dataset))
        
        return evaluation_dataset

    def evaluate_ragas_dataset(
            self,
            dataset: EvaluationDataset, 
            evaluator_llm: LangchainLLMWrapper, 
            project_name: str = None
    ) -> Dict[str, Any]:
        """Evaluate a RAGAS dataset with retriever-specific metrics
        
        Focus on retrieval quality metrics:
        - LLMContextRecall: Did we retrieve the reference documents?
        - ContextEntityRecall: Did we capture key entities?
        - NoiseSensitivity: Are we filtering irrelevant documents?
        
        Note: Generation metrics (Faithfulness, FactualCorrectness, ResponseRelevancy) 
        are excluded to reduce cost when comparing retrievers.
        """
        return evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(), 
                FactualCorrectness(), 
                ResponseRelevancy(),
                LLMContextRecall(),      # Primary retrieval metric
                ContextEntityRecall(),   # Entity coverage metric
                NoiseSensitivity(),      # Noise filtering metric
            ],
            llm=evaluator_llm,
            run_config=RunConfig(timeout=360),
            raise_exceptions=False,
        )

def get_langsmith_cost_stats(
    client: Client,
    project_name: str, 
    retriever_name: str = None,
    session_id: str = None,
) -> Dict[str, Any]:
    """Fetch pre-computed cost and latency statistics from LangSmith
    
    Args:
        project_name: LangSmith project name
        retriever_name: Optional filter for specific retriever runs (not used for project stats)
        session_id: Optional filter for specific evaluation session (not used for project stats)
    
    Returns:
        Dictionary with LangSmith's pre-computed statistics including p99, costs, tokens
    """
    filter_conditions = []
    
    if session_id:
        filter_conditions.append(f'has(tags, "session-{session_id}")')
    
    if retriever_name:
        filter_conditions.append(f'has(tags, "{retriever_name}")')
    
    filter_str = f'and({", ".join(filter_conditions)})' if filter_conditions else None
    
    # Get your tagged runs (the 10 question runs)
    runs = list(client.list_runs(
        project_name=project_name,
        run_type="chain",
        # filter='eq(name, "ragas evaluation")',
        # limit=100  # Get all matching runs
    ))
    
    logger.info(f"Found {len(runs)} tagged runs for {retriever_name}")
    
    if not runs:
        return {
            "total_cost": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "run_count": 0,
            "total_latency_seconds": 0,
            "avg_latency_seconds": 0,
        }
    
    # Get unique trace_ids from your runs
    trace_ids = set(run.trace_id for run in runs)
    
    # For each trace, find the root "ragas evaluation" run
    ragas_runs = []
    for trace_id in trace_ids:
        # logger.info(f"Processing trace_id: {trace_id}")
        trace_runs = list(client.list_runs(
            project_name=project_name,
            trace_id=trace_id,
            is_root=True
        ))
        ragas_runs.extend([r for r in trace_runs if r.name == "ragas evaluation" or r.name == "RunnableSequence"])
    
    logger.info(f"Found {len(ragas_runs)} ragas evaluation runs")
    # logger.info(ragas_runs)
    
    # Aggregate costs/latency from ragas evaluation runs
    total_cost = sum(r.total_cost for r in ragas_runs if r.total_cost)
    total_tokens = sum(r.total_tokens for r in ragas_runs if r.total_tokens)
    prompt_tokens = sum(r.prompt_tokens for r in ragas_runs if r.prompt_tokens)
    completion_tokens = sum(r.completion_tokens for r in ragas_runs if r.completion_tokens)
    
    latencies = [(r.end_time - r.start_time).total_seconds() 
                 for r in ragas_runs if r.end_time and r.start_time]
    
    return {
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "run_count": len(ragas_runs),
        "total_latency_seconds": sum(latencies),
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0,
    }

def main():
    evaluation_filepath = "evaluation/retriever"
    ragas_testset_filepath = "evaluation/financial_crimes_testset_ragas.jsonl"
    os.makedirs(evaluation_filepath, exist_ok=True)

    ragas_dataset = EvaluateRAGASDataset()

    logger.info("Create Qdrant vector store")
    vector_store_manager = QdrantStoreManager(path="./qdrant_data")
    vector_store = vector_store_manager.get_vector_store()

    # JSON Decoder errors in referenced_context field
    # ds = pd.read_csv(ragas_testset_filepath)
    with open(ragas_testset_filepath, 'r') as f:
        ds = []
        for row in f:
            ds.append(json.loads(row))
    rag_prompt = ragas_dataset.get_rag_prompt()
    chat_model = ragas_dataset.get_chat_model()

    retriever_map = {
        "naive_retriever": ragas_dataset.naive_retriever_chain(vector_store),
        # "bm25_retriever": bm25_retriever_chain(rag_prompt, chat_model, dataset),
        "contextual_compression_retriever": ragas_dataset.contextual_compression_retriever_chain(
            rag_prompt, chat_model, 
            ragas_dataset.naive_retriever_chain(vector_store)
        ),
        # "multi_query_retriever": multiquery_retriever_chain(rag_prompt, chat_model, 
        # naive_retriever_chain(rag_prompt, chat_model, vector_store)),
        # "parent_document_retriever": parent_document_retriever_chain(rag_prompt, chat_model, dataset),
        # "ensemble_retriever": ensemble_retriever_chain(rag_prompt, chat_model, 
        # [
        #     naive_retriever_chain(rag_prompt, chat_model, vector_store), 
        #     bm25_retriever_chain(rag_prompt, chat_model, dataset), 
        #     contextual_compression_retriever_chain(rag_prompt, chat_model, 
        #     naive_retriever_chain(rag_prompt, chat_model, vector_store)), 
        #     multiquery_retriever_chain(rag_prompt, chat_model, 
        #     naive_retriever_chain(rag_prompt, chat_model, vector_store)), 
        #     parent_document_retriever_chain(rag_prompt, chat_model, dataset)
        # ]),
    }
    # retriever_name = 'naive_retriever'
    # eval_ds = ragas_dataset.create_evaluation_dataset(
    #      ds, retriever_map[retriever_name], rag_prompt, chat_model,
    #                 retriever_name=retriever_name, session_id=EVALUATION_SESSION_ID
    # )
    
    # Use temperature=0 for deterministic evaluation
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    
    # Store results for comparison
    project_name = os.environ.get("LANGCHAIN_PROJECT", "Advanced-Retrieval-Evaluation")
    results_summary = []

    for retriever_name in retriever_map:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {retriever_name}")
        logger.info(f"{'='*60}")

        client = Client()
        # client.create_project()
        # Each retriever should have its own project to capture costs and latency by retriever
        project_retreiver_name = f"{project_name}-{retriever_name}"
        os.environ["LANGCHAIN_PROJECT"] = project_retreiver_name

        # logger.info(f"Set LANGCHAIN_PROJECT to: {os.environ['LANGCHAIN_PROJECT']}")
        logger.info(f"Expected project: {project_retreiver_name}")
        
        # Track timing
        start_time = time.time()

        logger.info(f"Generate evaluation dataset for `{retriever_name}`")

        with tracing_context(project_name=project_retreiver_name, client=client):
            eval_ds = ragas_dataset.create_evaluation_dataset(
                ds, retriever_map[retriever_name], rag_prompt, chat_model,
                    retriever_name=retriever_name, session_id=EVALUATION_SESSION_ID
                )

            logger.info(f"Evaluate `{retriever_name}` using RAGAS")
            eval_results = ragas_dataset.evaluate_ragas_dataset(eval_ds, evaluator_llm, project_retreiver_name)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Convert EvaluationResult to dict - only numeric columns
        results_df = eval_results.to_pandas()
        # Select only numeric columns for mean calculation
        numeric_cols = results_df.select_dtypes(include=['number']).columns
        results_dict = results_df[numeric_cols].mean().to_dict()
        
        # Store results
        results_summary.append({
            "retriever": retriever_name,
            "faithfulness": results_dict.get("faithfulness", 0),
            "factual_correctness": results_dict.get("factual_correctness", 0),
            "answer_relevancy": results_dict.get("answer_relevancy", 0),
            "context_recall": results_dict.get("context_recall", 0),
            "context_entity_recall": results_dict.get("context_entity_recall", 0),
            "noise_sensitivity": results_dict.get("noise_sensitivity_relevant", 0),
            "latency_seconds": latency,
        })
        
        logger.info(f"Results: {results_dict}")
        logger.info(f"Latency: {latency:.2f}s")
    
    # Fetch cost data from LangSmith
    logger.info(f"\n{'='*60}")
    logger.info("FETCHING COST DATA FROM LANGSMITH")
    logger.info(f"{'='*60}")
    
    try:
        client = Client()
        logger.info(f"Session ID: {EVALUATION_SESSION_ID}")
        for result in results_summary:
            retriever_name = result["retriever"]
            project_retreiver_name = f"{project_name}-{retriever_name}"
            try:
                cost_stats = get_langsmith_cost_stats(
                    client,
                    project_retreiver_name, 
                    retriever_name=retriever_name,
                    session_id=EVALUATION_SESSION_ID,
                )
                logger.info(cost_stats)
                result["langsmith_total_cost_usd"] = cost_stats["total_cost"]
                result["langsmith_total_tokens"] = cost_stats["total_tokens"]
                result["langsmith_llm_calls"] = cost_stats["run_count"]
                result["langsmith_total_latency"] = cost_stats["total_latency_seconds"]
                
                logger.info(f"{retriever_name}:")
                logger.info(f"  Total Cost: ${cost_stats['total_cost']:.4f}")
                logger.info(f"  Tokens: {cost_stats['total_tokens']:,}")
            except Exception as e:
                logger.warning(f"Could not fetch stats for {retriever_name}: {e}")
                result["langsmith_total_cost_usd"] = None
                result["langsmith_total_tokens"] = None
                result["langsmith_llm_calls"] = None
                result["langsmith_total_latency"] = None
    except Exception as e:
        logger.warning(f"Could not fetch LangSmith costs: {e}")
        logger.info("Continuing without cost data...")
    
    # Print summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'='*60}")
    
    summary_df = pd.DataFrame(results_summary)
    
    # Save results
    summary_df.to_csv(f"{evaluation_filepath}/retriever_evaluation_results.csv", index=False)
    logger.info(f"\nResults saved to: {evaluation_filepath}/retriever_evaluation_results.csv")
    logger.info("\nFor detailed cost analysis, check LangSmith dashboard at:")
    logger.info("https://smith.langchain.com/")

if __name__ == "__main__":
    main()