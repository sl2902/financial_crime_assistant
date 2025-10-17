"""Build Retriever using LangGraph"""
import os
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv
from typing_extensions import TypedDict
from loguru import logger

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, END

from src.ingestion.loader import QdrantStoreManager, vector_store_retriever

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Financial-Crimes-Assistant"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

class State(TypedDict):
  query: str
  context: List[Document]
  response: str

class FinancialCrimeRAGSystem:
    def __init__(
          self, 
          retriever,
          llm = None,
          relevance_threshold: float = 0.7):
      self.retriever = retriever
      self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
      self.relevance_threshold = relevance_threshold
      self.chat_prompt = self._create_prompt()
      self.graph = self._build_graph()
    
    def __repr__(self):
      return f"FinancialCrimeRAGSystem(llm={self.llm.model_name})"
   
    def _retrieve(self, state: State) -> Dict:
        """Retrieve chunks from Qdrant Vector Store"""
        retrieved_docs = self.retriever.invoke(state["query"])
        return {
            "context": retrieved_docs
        }
    
    def _generate(self, state: State) -> Dict:
        """Generate Response using LLM"""
        context = "\n\n".join(
           [
              f"Document {i+1} ({doc.metadata.get('lr_no', 'Unknown')}):\n{doc.page_content}"
              for i, doc in enumerate(state["context"])
           ]
           
        )
        generator_chain =  self.chat_prompt| self.llm | StrOutputParser()
        response = generator_chain.invoke(
            {
              "query" : state["query"], 
              "context" : context
            }
        )
        return {"response" : response}

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create Chat Prompt Template"""
        HUMAN_TEMPLATE = """
            You are an expert financial crime compliance assistant helping KYC analysts.

            Use the following context from SEC enforcement documents to answer the question.

            CONTEXT:
            {context}

            QUERY:
            {query}

            Instructions:
            - Answer based ONLY on the provided context
            - If the context doesn't contain the answer, say "I don't have enough information in the provided documents"
            - Cite specific cases, amounts, dates and location when relevant
            - Be concise but comprehensive

            ANSWER:
            """

        return ChatPromptTemplate.from_messages([
            ("human", HUMAN_TEMPLATE)
        ])
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(State)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")

        return workflow.compile()
    
    def query(self, question: str) -> str:
       """Main query method"""
       result = self.graph.invoke({
          "query": question,
          "context": [],
          "response": ""
       })

       return result["response"]

if __name__ == "__main__":
   
    store_manager = QdrantStoreManager(path="./qdrant_data")

    retriever = vector_store_retriever(store_manager)

    rag_system = FinancialCrimeRAGSystem(retriever=retriever)

    question = "What is securities fraud and what are the typical penalties?"
    answer = rag_system.query(question)
    
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}")

