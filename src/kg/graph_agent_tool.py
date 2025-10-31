"""Compact graph query tool for agent integration using NL-to-Cypher"""

import json
from loguru import logger
from typing import Optional, Dict, List, Any

from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel

from src.kg.neo4j_manager import Neo4jManager
from src.prompts.prompts import neo4j_prompt

class GraphAgentTool:
    """Compact graph query tool using LLM to generate Cypher queries"""

    def __init__(self, neo4j_manager: Neo4jManager, llm: BaseChatModel):
        """Initialize graph agent tool
        
        Args:
            neo4j_manager: Neo4jManager instance
            llm: LangChain LLM instance
        """
        self.manager = neo4j_manager
        self.llm = llm
        self.schema = self._get_schema_description()
        self._last_query_results = {}
    
    def _get_schema_description(self) -> str:
        return """
            Neo4j Knowledge Graph Schema:

            NODES:
            1. Person
            Properties: name (string), role (string)
            
            2. Company
            Properties: name (string), ticker (string), industry (string), description (string)
            
            3. Case
            Properties: lr_number (string, unique), title (string), crime_types (list of strings),
                        filing_date (date), summary (string), url (string)
            
            4. Penalty
            Properties: id (string, unique), penalty_type (string), amount (float),
                        description (string), recipient (string)

            RELATIONSHIPS:
            1. (Person)-[:CHARGED_IN]->(Case)
            - Person was charged in a case
            
            2. (Person)-[:WORKED_AT]->(Company)
            - Person worked at or was affiliated with company
            
            3. (Person)-[:RECEIVED_PENALTY]->(Penalty)
            - Person received a penalty
            
            4. (Company)-[:INVOLVED_IN]->(Case)
            - Company was involved in a case
            
            5. (Company)-[:RECEIVED_PENALTY]->(Penalty)
            - Company received a penalty
            
            6. (Case)-[:HAS_PENALTY]->(Penalty)
            - Case resulted in this penalty

            COMMON QUERIES:
            - Find Ponzi schemes: WHERE 'Ponzi Scheme' IN c.crime_types
            - Find high penalties: WHERE p.amount > 10000000
            - Find person connections: MATCH (p:Person)-[:WORKED_AT]->(c:Company)<-[:WORKED_AT]-(other:Person)
            - Find repeat offenders: MATCH (p:Person)-[:CHARGED_IN]->(c:Case) WITH p, count(c) AS cases WHERE cases > 1
            - Date filtering: WHERE c.filing_date >= date('2023-01-01')
            - Date ranges: WHERE c.filing_date >= date('2024-01-01') AND c.filing_date <= date('2024-12-31')
            - After date: WHERE c.filing_date > date('2025-01-01')
            - Before date: WHERE c.filing_date < date('2023-12-31')
            - Case-insensitive search: WHERE toLower(p.name) CONTAINS toLower('smith')
            """
    
    def _nl_to_cypher(self, question: str) -> str:
        """Convert natural language question to Cypher query using LLM
        
        Args:
            question: Natural language question
            
        Returns:
            Cypher query string
        """
        prompt = neo4j_prompt.format(schema=self.schema, question=question)
        
        try:
            response = self.llm.invoke(prompt)
            cypher = response.content.strip()

            cypher = cypher.replace("```cypher query", "").replace("```cypher", "").replace("```", "").strip()
            logger.info(f"Generated Cypher: {cypher}")
            return cypher
        except Exception as e:
            logger.error(f"Error generating Cypher: {e}")
            return None
    
    def _validate_cypher(self, cypher: str) -> bool:
        """Strict validation of Cypher query to prevent destructive operations
        
        Args:
            cypher: Cypher query string
            
        Returns:
            True if query appears safe and valid
        """
        cypher_lower = cypher.lower().strip()

        if 'match' not in cypher_lower and 'return' not in cypher_lower:
            logger.warning("Query validation failed: No MATCH or RETURN found")
            return False
        
        dangerous_keywords = [
            'delete',
            'remove',
            'detach delete',
            'detach',
            'drop',
            'create ',  # Space after to avoid matching "OPTIONAL MATCH"
            'create(',
            'merge',
            'set ',
            'set(',
            'alter',
            'truncate',
            'grant',
            'revoke',
            'create constraint',
            'drop constraint',
            'create index',
            'drop index',
        ]
        
        for keyword in dangerous_keywords:
            if keyword in cypher_lower:
                logger.error(f"Query validation failed: Dangerous keyword '{keyword}' detected")
                return False
        
        # Additional safety: Check for suspicious patterns
        suspicious_patterns = [
            'call {',  # Subqueries that might do writes
            'call apoc',  # APOC procedures can be dangerous
            'foreach',  # Can be used for writes
        ]

        for pattern in suspicious_patterns:
            if pattern in cypher_lower:
                logger.warning(f"Query validation warning: Suspicious pattern '{pattern}' detected")
                return False
        
        # Must be a reasonable length (prevent injection attempts)
        if len(cypher) > 2000:
            logger.warning("Query validation failed: Query too long")
            return False
        
        return True
    
    def _execute_query(self, cypher: str) -> Dict[str, Any]:
        """Execute Cypher query and return results
        
        Args:
            cypher: Cypher query string
            
        Returns:
            Dictionary with results
        """
        try:
            if not self._validate_cypher(cypher):
                return {
                    "error": "Invalid or unsafe Cypher query",
                    "cypher": cypher
                }
            results = self.manager.execute_query(cypher)

            return {
                "success": True,
                "count": len(results),
                "results": results,
                "cypher": cypher
            }
        except Exception as e:
            logger.error(f"Error executing Cypher: {e}")
            return {
                "error": str(e),
                "cypher": cypher
            }
    
    def _format_results_for_agent(self, results: Dict[str, Any], limit: int = 10) -> str:
        """Format query results as text for agent context
        
        Args:
            results: Query results dictionary
            limit: Limit number of results for context
            
        Returns:
            Formatted string
        """
        if "error" in results:
            return f"Graph query error: {results['error']}"
        
        if not results.get("results"):
            return "Knowledge graph returned no results for this query."
        
        output = ["KNOWLEDGE GRAPH RESULTS:\n"]

        for i, record in enumerate(results["results"][:limit], start=1):
            record_lines = [f"\n{i}. "]

            for key, value in record.items():
                if value and key != "id":
                    if isinstance(value, List):
                        if value:
                            value_str = ", ".join(str(v) for v in value if v)
                            if value_str:
                                record_lines.append(f"{key}: {value_str}")
                    elif isinstance(value, (int, float)) and key in ['amount', 'total_penalty', 'total_penalties']:
                        record_lines.append(f"{key}: ${value:,.0f}")
                    else:
                        record_lines.append(f"{key}: {value}")
            
            if len(record_lines) > 1:  # Only add if there's content
                output.append("; ".join(record_lines))
        
        return "\n".join(output)
    
    def search(self, question: str) -> str:
        """Main search method for agent tool
        
        Args:
            question: Natural language question
            
        Returns:
            Formatted results string or error message
        """
        logger.info(f"Graph tool query: {question}")

        cypher = self._nl_to_cypher(question)

        if cypher is None:
            return ("I couldn't generate a valid graph query. "
            "Please rephrase your question to focus on relationships between people, companies, "
            "and cases in the financial crime database."
            )
        
        results = self._execute_query(cypher)

        if "error" in results:
            error_msg = results["error"]
            logger.error(f"Graph query error: {error_msg}")
            return f"Graph query failed: {error_msg}. Please try rephrasing your question or asking about specific cases, people, or companies."
        
        self._last_query_results = results.copy()
        logger.debug(f"Stored {results.get('count', 0)} results for visualization")
        
        formatted = self._format_results_for_agent(results)
        
        # logger.info(f"Graph tool returning {results.get('count', 0)} results")
        
        return formatted
    
    def get_results_for_visualization(self) -> List[Dict[str, Any]]:
        """Get raw Neo4j results for graph visualization
        
        Returns:
            List of Neo4j result dictionaries suitable for pyvis visualization
        """
        return self.last_results.get('results', [])
    
    def clear_results(self):
        """Clear stored results (call after visualization to free memory)"""
        self.last_results = {'results': [], 'count': 0}


def create_graph_tool(
        neo4j_manager: Neo4jManager,
    llm: BaseChatModel,
    tool_name: str = "search_knowledge_graph",
    tool_description: Optional[str] = None
) -> Tool:
    """Factory function to create a graph tool for agent.
    
    Args:
        neo4j_manager: Neo4jManager instance
        llm: LangChain LLM instance
        tool_name: Name for the tool
        tool_description: Optional custom description
        
    Returns:
        LangChain Tool instance
    
    Example:
        >>> from src.kg import Neo4jManager
        >>> from langchain_openai import ChatOpenAI
        >>> 
        >>> manager = Neo4jManager()
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> graph_tool = create_graph_tool(manager, llm)
        >>> 
        >>> # Add to agent tool belt
        >>> tool_belt = [rag_tool, tavily_tool, graph_tool]
    """
    if tool_description is None:
        tool_description = """Query the knowledge graph for RELATIONSHIPS and CONNECTIONS.

            USE THIS TOOL WHEN the query contains:
            - WHO questions: "Who was charged in [crime type]?"
            - WHICH questions: "Which companies were involved?"
            - CONNECTION words: "show connections for [person]"
            - LIST/SHOW requests: "List all people in [case type]"
            
            ⚠️ CRITICAL CITATION REQUIREMENTS:
            Every factual claim MUST be cited using markdown link format:
            [CASE_NUMBER](URL)
            
            Examples:
            - Jed Wood was charged in [LR-26415](https://www.sec.gov/litigation/litreleases/lr-26415).
            - Wood worked at Agridime, LLC [LR-26415](https://www.sec.gov/litigation/litreleases/lr-26415).
            
            Rules:
            - Put citation at the END of the sentence before the period
            - Link text: Case LR number where the relationship is established
            - Link URL: Full SEC case URL
            - NEVER make claims without citations
            
            The tool returns case URLs with each result - always use them.
            
            NEVER use for:
            - "What happened?" (use search_sec_documents)
            - Current news (use tavily_search_results_json)
            
            Returns:
                Dictionary with entities and their associated case LR numbers and URLs
    """
    
    graph_agent = GraphAgentTool(neo4j_manager, llm)

    tool = Tool(
        name=tool_name,
        func=graph_agent.search,
        description=tool_description
    )
    tool._search_instance = graph_agent

    return tool

if __name__ == "__main__":
    
    from langchain_openai import ChatOpenAI
    
    print("=" * 70)
    print("GRAPH AGENT TOOL TEST")
    print("=" * 70)
    
    # Initialize
    manager = Neo4jManager()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create tool
    graph_tool = create_graph_tool(manager, llm)
    
    # Test queries
    test_queries = [
        "Show me all Ponzi schemes",
        "Who worked with Jed Wood?",
        "Find cases with penalties over $50 million",
        "What companies have been involved in multiple fraud cases?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('='*70)
        
        result = graph_tool.func(query)
        print(result)
        
        if i < len(test_queries):
            input("\nPress Enter for next query...")
    
    print("\n" + "="*70)
    print(" All tests complete!")
    print("="*70)

        