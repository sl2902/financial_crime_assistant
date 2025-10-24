"""Build RAG Retriever using LangGraph"""
import os
import re
import json
from typing import Any, List, Dict, Tuple, Optional
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from loguru import logger
import asyncio

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

from qdrant_client.models import (
    Filter, 
    FieldCondition, 
    MatchValue, 
    MatchAny, 
    Range
)

from src.ingestion.loader import QdrantStoreManager, vector_store_retriever
from src.schemas.rag_schemas import(
   Citation,
   StructuredAnswer, 
   RAGResponse
) 

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
            - Format your answer with clear structure:
               * Use bullet points for listing multiple cases or facts
               * Use numbered lists for comparisons or sequences
               * Use headers (##) for different sections if the query asks for comparison
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
  limit: int

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
      self.structured_llm = self.llm.with_structured_output(StructuredAnswer)

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
            logger.info(f"Number of filtered results for `Agentic RAG` {len(results)}")
            self.last_retrieved_docs = [
               Document(
                  page_content=point.payload.get("page_content", ""),
                  metadata=point.payload.get("metadata", {})
               )
               for point in results
         ]
        
         else:
            self.last_retrieved_docs  = self.retriever.invoke(query)
         
         formatted_chunks = []
         for i, doc in enumerate(self.last_retrieved_docs, 1):
            lr_no = doc.metadata.get('lr_no', 'Unknown')
            url = doc.metadata.get('url', 'N/A')
            title = doc.metadata.get('title', 'SEC Document')
            
            chunk_with_citation = f"""Source {i}: {lr_no} - {title}
            Citation format to use: [{lr_no}]({url})

            {doc.page_content}

                  ---"""
            formatted_chunks.append(chunk_with_citation)
    
         return "\n\n".join(formatted_chunks)
         # return "\n\n".join([doc.page_content for doc in self.last_retrieved_docs ])
         
      self.rag_tool = Tool(
         name="search_sec_documents",
         func=search_sec_documents,
         description="""Search SEC enforcement documents about financial crimes
         Use this tool for ANY questions about:
         - Penalties, fines, disgorgement amounts
         - Financial crime cases and enforcement actions
         - Historical SEC cases and outcomes
         - Specific case details by crime type or date
         Always use this tool unless the query explicitly asks for current news."""
      )
      self.tool_belt = [self.tavily_search_tool, self.rag_tool]
      self.llm_bind_tool = self.llm.bind_tools(self.tool_belt)
   
   def __repr__(self):
      return f"FinancialCrimeRAGSystem(llm={self.llm.model_name})"
   
   def _retrieve(self, state: State) -> Dict:
      """Retrieve chunks from Qdrant Vector Store"""
      qdrant_filter = state.get("filter")
      query = state.get("query")
      limit = state.get("limit")
   
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
               limit=limit
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
               logger.info(f"Doc lr_no: {metadata.get('lr_no')}, Penalty category: {metadata.get('penalty_category')}, content length: {len(page_content)}")
               
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
         retrieved_docs = self.retriever.invoke(query)[:limit]
      
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
      if len(messages) == 1:
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=self.agent_system_prompt)] + messages
      if len(messages) > 2 and any(hasattr(m, 'type') and m.type == 'tool' for m in messages):
         # Use structured output for final answer
         response = self.structured_llm.invoke(messages)
         # Convert Pydantic model to string for messages
         response = AIMessage(content=response.answer_text)
      else:
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
      self.agent_system_prompt = """You are an expert financial crime compliance assistant.

      ANSWER FORMAT:
      - Use clear section headers with markdown (##, ###)
      - Use bullet points for lists of cases or penalties
      - Use numbered lists for step-by-step comparisons
      - Keep paragraphs concise

      CITATION RULES:
      FOR SEC DOCUMENTS (from search_sec_documents tool):
      - Inline citations: [LR-XXXXX](URL)
      - Place immediately after factual claims
      - Format: "The defendant was fined $500,000 [LR-26123](https://sec.gov/...)."

      FOR WEB SOURCES (from tavily_search tool):
      - Use website name: [Website Name](URL)
      - Format: "Recent reports indicate... [Reuters](https://reuters.com/...)."
      - If no clear name, use: [Read more](URL)

      Never use generic text like "(source)" - always use specific source names or LR numbers."""
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
    
   def query(self, question: str, filter: List[FieldCondition] = None, limit: int = 3) -> str:
      """Main query method"""
      lr_values = re.findall(r"lr-\d+", question.lower())
      logger.info(f"RAG query has {lr_values} case numbers in the query '{question}'")

      qdrant_filter = None
      if lr_values:
         qdrant_filter = filter_by_lr_number(lr_values)
      
      if filter:
         logger.info(f"Applying `Plain RAG` filter")
      
      combined_filter = None
      if qdrant_filter and filter:
         combined_filter = qdrant_filter + filter
      elif qdrant_filter:
         combined_filter = qdrant_filter
      else:
         combined_filter = filter
      
      logger.info(f"Overall filter condition being applied to `Plain RAG` {combined_filter}")

      result = self.graph.invoke({
            "query": question,
            "context": [],
            "response": "",
            "filter": combined_filter,
            "limit": limit
         }
      )

      return result["response"]
    
   def agent_query(self, question: str, filter: Filter = None) -> dict:
      """Query method for RAG Agent"""
      lr_values = re.findall(r"lr-\d+", question.lower())
      logger.info(f"Agentic RAG query has {lr_values} case numbers in the query '{question}'")

      lr_filter, metadata_filter = None, None
      if lr_values:
         lr_filter = filter_by_lr_number(lr_values)
      
      if filter:
         metadata_filter = filter
         logger.info(f"Applying `Agentic RAG` filter {metadata_filter}")
      
      combined_filter = None
      if lr_filter and metadata_filter:
         combined_filter = lr_filter + metadata_filter
      elif lr_filter:
         combined_filter = lr_filter
      else:
         combined_filter = metadata_filter
      
      self.current_filter = combined_filter
      logger.debug(f'Is current filter set? {self.current_filter}')

      inputs = {"messages": [HumanMessage(content=question)]}
      result = self.agent_graph.invoke(inputs)

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
                     retrieved_docs.extend(self.last_retrieved_docs)
                     # if self.current_filter:
                     #       docs = self.last_retrieved_docs
                     # else:
                     #       docs = self.retriever.invoke(rag_query)
                     # retrieved_docs.extend(docs)
      
      self.current_filter = None
      self.last_retrieved_docs = []
      
      return {
         "result": answer,
         "tools_used": list(set(tools_used)),
         "sources": retrieved_docs,  # Return deduplicated sources
         "rag_query": rag_query or '',
      }
   
   def _add_citations(self, answer: str, docs: List[Document]) -> str:
      """Append a sources section with proper case names and URLs"""
      citations = "\n\n---\n**Sources:**\n"
      for i, doc in enumerate(docs, 1):
         lr_no = doc.metadata.get('lr_no', 'Unknown')
         url = doc.metadata.get('url', 'N/A')
         title = doc.metadata.get('title', 'Unnamed SEC Case')
         citations += f"- [{lr_no}]({url}) — {title}\n"
      return answer + citations
   
   def _add_citations_with_llm(self, answer: str, sources: List[Document]) -> str:
         """Use LLM to add inline citations with proper markdown links"""
         
         # Build numbered sources with full details
         sources_list = []
         for i, doc in enumerate(sources, 1):
            lr_no = doc.metadata.get('lr_no', 'Unknown')
            url = doc.metadata.get('url', 'N/A')
            title = doc.metadata.get('title', 'SEC Document')[:60]
            sources_list.append(f"Source [{i}]: {lr_no} - {title}\nURL: {url}")
         
         sources_text = "\n\n".join(sources_list)
         
         prompt = f"""Add inline citation links in MARKDOWN format to this answer.

            CRITICAL RULES: 
            1. Format citations as: [Source N](URL) where N is the source number
            2. Place citation at the END of the sentence: "The penalty was $500,000 [Source 1](https://sec.gov/...)."
            3. Use the EXACT URL from the sources below
            4. Match facts to the correct source - read the source content carefully
            5. If a fact appears in multiple sources, cite the most specific one
            6. Don't cite general statements or your own analysis

            ANSWER TO ADD CITATIONS TO:
            {answer}

            AVAILABLE SOURCES:
            {sources_text}

            Return the complete answer with markdown citation links [Source N](URL) added after factual claims:"""
         
         cited = self.llm.invoke(prompt).content
         
         # Add clean sources section
         sources_section = "\n\n---\n**Sources:**\n"
         for i, doc in enumerate(sources, 1):
            lr_no = doc.metadata.get('lr_no', 'Unknown')
            url = doc.metadata.get('url', 'N/A')
            title = doc.metadata.get('title', 'SEC Document')[:60]
            sources_section += f"{i}. [{lr_no}]({url}) — {title}\n"
         
         return cited + sources_section
   
   def _reformat_to_bullets(self, answer: str) -> str:
         """Reformat prose answer to bullet points"""
         prompt = f"""Reformat this answer using markdown bullets and clear structure.

            Rules:
            - Use ## for main sections (Historical, Current, Comparison)
            - Use * for each case/point
            - Keep all citations intact: [LR-XXXXX](url) and [Source](url)
            - Format penalties as sub-bullets with amounts

            Answer to reformat:
            {answer}

            Reformatted answer:"""
         
         return self.llm.invoke(prompt).content

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

def extract_filters(question: str) -> Optional[Filter | None]:
   """Extract filter criteria from question"""
   conditions = []
   
   # LR number filter
   lr_values = re.findall(r"lr-\d+", question.lower())
   if lr_values:
      conditions.extend([
         FieldCondition(key="metadata.lr_no", match=MatchValue(value=lr.upper()))
         for lr in lr_values
      ])
   
   # Crime type filter
   crime_types = {
      "ponzi": "Ponzi Scheme",
      "insider trading": "Insider Trading",
      "securities fraud": "Securities Fraud",
      "market manipulation": "Market Manipulation"
   }
   for keyword, crime_type in crime_types.items():
      if keyword in question.lower():
         conditions.append(
               FieldCondition(key="metadata.crime_type", match=MatchValue(value=crime_type))
         )
   
   # Date range filter
   year_match = re.search(r"\b(202[3-5])\b", question)
   if year_match:
      year = year_match.group(1)
      conditions.append(
         FieldCondition(
               key="metadata.date",
               range=Range(
                  gte=f"{year}-01-01",
                  lte=f"{year}-12-31"
               )
         )
      )
   
   if conditions:
      return Filter(should=conditions) if len(conditions) == 1 else Filter(must=conditions)
   return None

if __name__ == "__main__":
   
   store_manager = QdrantStoreManager(path="./qdrant_data")

   retriever = vector_store_retriever(store_manager)

   rag_system = FinancialCrimeRAGSystem(retriever=retriever)

   # question = "What is securities fraud and what are the typical penalties?"
   # answer = rag_system.query(question)
   
   # uses sec_documents
   question = "What penalties were issued for insider trading in 2025?"
   # question = "What is LR-26161 about?" #"Who is Baris Cabalar? Check documents"

   # answer = rag_system.query(question)

   # answer = rag_system.agent_query(question)

   # use Tavily
   # question = "What financial crime news happened this week?"

   answer = rag_system.agent_query(question)

   logger.info(f"Question: {question}")
   logger.info(f"Answer: {answer}")

