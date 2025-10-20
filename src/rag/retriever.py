"""Build RAG Retriever using LangGraph"""
import os
import re
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from loguru import logger
import asyncio

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain.tools import Tool
from langchain_core.tools import (
    tool,
)

from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from qdrant_client.models import Filter, FieldCondition, MatchValue

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
            - When citing sources, use the format: [LR-XXXXX](URL) instead of just "Document 1"
            - Include the full SEC.gov URL for each citation
            - If the context doesn't contain the answer, say "I don't have enough information in the provided documents"
            - Cite specific cases, amounts, dates and location when relevant
            - When mentioning monetary amounts, ALWAYS include the dollar sign ($) - for example: "$6.76 billion" not "6.76 billion"
            - Be concise but comprehensive

            ANSWER:
            """

class State(TypedDict):
  query: str
  context: List[Document]
  response: str
  filter: Any

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

class FinancialCrimeRAGSystem:
   def __init__(
         self, 
         retriever,
         llm = None,
         collection_name: str = "financial_crimes",
         relevance_threshold: float = 0.7):
      self.collection_name = collection_name
      self.retriever = retriever
      self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
      self.relevance_threshold = relevance_threshold
      self.chat_prompt = self._create_prompt()
      self.tavily_search_tool = TavilySearch(max_results=5)
      self.graph = self._build_graph()
      self.agent_graph = self._build_agent_graph()
      self.current_filter = None
      self.last_retrieved_docs = []

      def search_sec_documents(query: str, search_k : int = 3) -> str:
         """Search SEC enforcement documents about financial crimes, fraud cases, and penalties"""
         if self.current_filter:
            logger.info(f"RAG tool using filter")
            vector_store = self.retriever.vectorstore
            # retrieved_docs = vector_store.similarity_search(
            #    query=query,
            #    k=search_k,
            #    filter=self.current_filter
            # )
            results, _ = self.retriever.vectorstore.client.scroll(
               collection_name=self.collection_name,
               scroll_filter=self.current_filter,
               limit=search_k,
               with_payload=True
            )
            self.last_retrieved_docs = [
               Document(
                  page_content=point.payload.get("page_content", ""),
                  metadata=point.payload.get("metadata", {})
               )
               for point in results
         ]
        
         else:
            self.last_retrieved_docs  = self.retriever.invoke(query)
         return "\n\n".join([doc.page_content for doc in self.last_retrieved_docs ])
         
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
      qdrant_filter = state.get("filter")
      query = state.get("query")
   
      if qdrant_filter:
         logger.info("USING FILTER SEARCH")
         
         vector_store = self.retriever.vectorstore

         embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
         query_embedding = embeddings.embed_query(query)
         
         results = vector_store.client.query_points(
               collection_name=self.collection_name,
               query=query_embedding,
               query_filter=qdrant_filter,
               with_payload=True,
               limit=5
         )

         retrieved_docs = []
         for point in results.points:
               # Check the payload structure
               # payload = result.payload
               payload = point.payload
               
               # The page_content is at top level
               page_content = payload.get("page_content", "")
               
               # Metadata is nested
               metadata = payload.get("metadata", {})
               
               # Log for debugging
               logger.info(f"Doc lr_no: {metadata.get('lr_no')}, content length: {len(page_content)}")
               
               if page_content:  # Only add if we have content
                  doc = Document(
                     page_content=page_content,
                     metadata=metadata
                  )
                  retrieved_docs.append(doc)
               else:
                  logger.warning("Found result but no page_content!")
         
         logger.info(f"Filter search returned {len(results.points)} results")
      else:
         logger.info("USING NORMAL SEARCH")
         retrieved_docs = self.retriever.invoke(query)
      
      return {
            "context": retrieved_docs
      }

   
   def _generate(self, state: State) -> Dict:
      """Generate Response using LLM"""
      context = "\n\n".join(
         [
            f"Document {i+1} ({doc.metadata.get('lr_no', 'Unknown')})\n"
            f"Source: {doc.metadata.get('url', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for i, doc in enumerate(state["context"])
         ]
         
      )
      generator_chain =  (
         self.chat_prompt
         | self.llm 
         | StrOutputParser()
      )
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
      lr_values = re.findall(r"lr-\d+", question.lower())
      logger.info(f"RAG query has {lr_values} case numbers in the query '{question}'")

      qdrant_filter = None
      if lr_values:
         qdrant_filter = filter_by_lr_number(lr_values)

      result = self.graph.invoke({
            "query": question,
            "context": [],
            "response": "",
            "filter": qdrant_filter
         }
      )

      return result["response"]
    
   def agent_query(self, question: str) -> str:
      """Query method for RAG Agent"""
      lr_values = re.findall(r"lr-\d+", question.lower())
      logger.info(f"Agentic RAG query has {lr_values} case numbers in the query '{question}'")

      if lr_values:
         self.current_filter = filter_by_lr_number(lr_values)

      inputs = {"messages": [HumanMessage(content=question)]}
      result = self.agent_graph.invoke(
         inputs
      )

      answer = result["messages"][-1].content

      tools_used = []
      retrieved_docs = []
      rag_query = ""

      for msg in result["messages"]:
         if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
               tool_name = tool_call.get("name", "")
               tools_used.append(tool_name)

               if tool_name == "search_sec_documents":
                  args = tool_call.get('args', {})
                  rag_query = (
                        args.get('query') or 
                        args.get('__arg1') or 
                        args.get('input') or
                        question
                     )
                  if self.current_filter:
                     docs = self.last_retrieved_docs
                  else:
                     docs = self.retriever.invoke(rag_query)
                  retrieved_docs.extend(docs)
      
      if retrieved_docs and 'search_sec_documents' in tools_used:
         logger.info(f'Add citation to {question}')
         answer = self._add_citations(answer, retrieved_docs)
      
      self.current_filter = None
      self.last_retrieved_docs = []
      
      return {
         "result": answer,
         "tools_used": list(set(tools_used)),
         "sources": retrieved_docs,
         "rag_query": rag_query or '',
      }
   
   def _add_citations(self, answer: str, docs: List[Document]) -> str:
      """Add source citations to answer"""
      citations = "\n\n---\n**Sources:**\n"
      for i, doc in enumerate(docs, 1):
         lr_no = doc.metadata.get('lr_no', 'Unknown')
         url = doc.metadata.get('url', 'N/A')
         title = doc.metadata.get('title', 'SEC Document')[:80]
         query = doc.metadata.get('rag_query', '')
         citations += f"{i}. [{lr_no}]({url}) {query} - {title}...\n"
      
      return answer + citations

def filter_by_lr_number(lr_nos: List[str]) -> FieldCondition:
   """Filter Qdrant using LR number"""
   return Filter(
            should=[
                FieldCondition(
                    key="metadata.lr_no",
                    match=MatchValue(value=lr.upper())
                )
                for lr in lr_nos
            ]
        )

if __name__ == "__main__":
   
   store_manager = QdrantStoreManager(path="./qdrant_data")

   retriever = vector_store_retriever(store_manager)

   rag_system = FinancialCrimeRAGSystem(retriever=retriever)

   # question = "What is securities fraud and what are the typical penalties?"
   # answer = rag_system.query(question)
   
   # uses sec_documents
   question = "What penalties were issued for insider trading in 2025?"
   question = "What is LR-26161 about?" #"Who is Baris Cabalar? Check documents"

   # answer = rag_system.query(question)

   # answer = rag_system.agent_query(question)

   # use Tavily
   # question = "What financial crime news happened this week?"

   answer = rag_system.agent_query(question)

   logger.info(f"Question: {question}")
   logger.info(f"Answer: {answer}")

