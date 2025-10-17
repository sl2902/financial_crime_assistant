"""Build RAG Retriever using LangGraph"""
import os
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from loguru import logger
import asyncio

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain.tools import Tool
from langchain_core.tools import (
    tool,
)

from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.ingestion.loader import QdrantStoreManager, vector_store_retriever

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Financial-Crimes-Assistant"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

CHAT_PROMPT = """
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

class State(TypedDict):
  query: str
  context: List[Document]
  response: str

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

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
      self.tavily_search_tool = TavilySearch(max_results=5)
      self.graph = self._build_graph()
      self.agent_graph = self._build_agent_graph()

      def search_sec_documents(query: str, search_k : int = 3) -> str:
        """Search SEC enforcement documents about financial crimes, fraud cases, and penalties"""
        retrieved_docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in retrieved_docs[:search_k]])
      
      self.rag_tool = Tool(
         name="search_sec_documents",
         func=search_sec_documents,
         description="Search SEC enforcement documents about financial crimes"
      )
      self.tool_belt = [self.tavily_search_tool, self.rag_tool]
      self.llm_bind_tool = self.llm.bind_tools(self.tool_belt)
    
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
    
    def _call_agent_model(self, state: AgentState) -> Dict:
        """Call Agent node"""
        messages = state["messages"]
        response = self.llm_bind_tool.invoke(messages)
        
        return {
           "messages" : [response]
        }
    
    def _tool_call(self, state: AgentState) -> ToolNode:
       """Initialize ToolNode"""
       return ToolNode(self.tool_belt)

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create Chat Prompt Template"""
        HUMAN_TEMPLATE = CHAT_PROMPT

        return ChatPromptTemplate.from_messages([
            ("human", HUMAN_TEMPLATE)
        ])
    
    def _build_graph(self):
        """Build a basic RAG LangGraph workflow"""
        workflow = StateGraph(State)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")

        return workflow.compile()
    
    def _build_agent_graph(self):
       """Build an Agent RAG LangGraph workflow"""
       workflow = StateGraph(AgentState)

       workflow.add_node("agent", self._call_agent_model)
       workflow.add_node("action", self._tool_call)
       workflow.set_entry_point("agent")
       workflow.add_edge("action", "agent")
       workflow.add_conditional_edges("agent", self._should_continue)

       return workflow.compile()
    
    def _convert_inputs(self, input_object: Dict):
       """Preprocess user query"""
       return {"messages" : [HumanMessage(content=input_object["text"])]}

    def _parse_output(self, input_state: Dict):
       """Parse LLM output"""
       return {"answer" : input_state["messages"][-1].content}
    
    def _generate_agent_response(self):
       """Prepare Agent LCEL Chain"""
       agent_chain_with_formatting = self._convert_inputs | self._build_agent_graph | self._parse_output

       return agent_chain_with_formatting
    
    def _should_continue(self, state: AgentState):
        """Conditional routing"""
        last_message = state["messages"][-1]

        if last_message.tool_calls:
          return "action"
        
        return END
    
    # async def agent_query(self, inputs: Dict[str, Any]) -> str:
    #     """Query method for RAG Agent with streaming"""
    #     final_response = None
        
    #     async for chunk in self.agent_graph.astream(inputs, stream_mode="updates"):
    #         for node, values in chunk.items():
    #             logger.info(f"Receiving update from node: '{node}'")
    #             logger.info(values["messages"])
    #             final_response = values["messages"][-1].content
    
    #     return final_response
    
    def query(self, question: str) -> str:
       """Main query method"""
       result = self.graph.invoke({
          "query": question,
          "context": [],
          "response": ""
       })

       return result["response"]
    
    def agent_query(self, question: str) -> str:
       """Query method for RAG Agent"""
       inputs = {"messages": [HumanMessage(content=question)]}
       result = self.agent_graph.invoke(
          inputs
       )

       return result["messages"][-1].content

if __name__ == "__main__":
   
    store_manager = QdrantStoreManager(path="./qdrant_data")

    retriever = vector_store_retriever(store_manager)

    rag_system = FinancialCrimeRAGSystem(retriever=retriever)

    # question = "What is securities fraud and what are the typical penalties?"
    # answer = rag_system.query(question)
    
    # uses sec_documents
    question = "What penalties were issued for insider trading in 2025?"

    # answer = rag_system.agent_query(question)

    # use Tavily
    question = "What financial crime news happened this week?"

    answer = rag_system.agent_query(question)

    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}")

    test_queries = [
    "What is securities fraud and what are typical penalties?",
    "What penalties were issued for insider trading in 2025?",
    "Tell me about Ponzi scheme cases",
    "What is the difference between securities fraud and wire fraud?",
]

