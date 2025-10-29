"""Compact graph query tool for agent integration using NL-to-Cypher"""

import json
from loguru import logger
from typing import Optional, Dict, List, Any

from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel

from src.kg.neo4j_manager import Neo4jManager

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
        self.last_query_results = {}
    
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
            - Case-insensitive search: WHERE toLower(p.name) CONTAINS toLower('smith')
            """
    
    def _nl_to_cypher(self, question: str) -> str:
        """Convert natural language question to Cypher query using LLM
        
        Args:
            question: Natural language question
            
        Returns:
            Cypher query string
        """
        prompt = f"""You are a Neo4j Cypher query expert. Convert the natural language question into a valid Cypher query.

            SCHEMA:
            {self.schema}

            RULES:
            1. Return ONLY the Cypher query, no explanations
            2. Use MATCH, WHERE, RETURN statements appropriately
            3. Include OPTIONAL MATCH for related data that might not exist
            4. Use LIMIT 20 by default unless the question specifies a different number
            5. For text searches, use: toLower(property) CONTAINS toLower($search_term)
            6. For date filtering, use: date() function
            7. Always order by filing_date DESC for cases
            8. Use collect() and count() for aggregations
            9. For amount comparisons, ensure penalty.amount IS NOT NULL
            10. Return meaningful property names in results
            11. When using collect() or aggregations, ORDER BY must use the alias from RETURN, not the original variable

            IMPORTANT PATTERNS:
            - Ponzi schemes: WHERE 'Ponzi Scheme' IN c.crime_types
            - Person search: WHERE toLower(p.name) CONTAINS toLower('...')
            - High penalties: WHERE pen.amount > 10000000 AND pen.amount IS NOT NULL
            - Recent cases: ORDER BY c.filing_date DESC LIMIT 10
            - Co-defendants: (p1:Person)-[:CHARGED_IN]->(case:Case)<-[:CHARGED_IN]-(p2:Person) WHERE p1 <> p2

            === CRITICAL OUTPUT FORMAT RULES ===

            When returning results, you MUST use these exact aliases:

            1. Person data MUST use alias 'persons' (plural, as list):
            ✅ CORRECT: collect(DISTINCT p.name) AS persons
            ❌ WRONG: p.name AS defendant
            ❌ WRONG: p.name AS person_name
            ❌ WRONG: collect(p.name) AS defendants

            2. Company data MUST use alias 'companies' (plural, as list):
            ✅ CORRECT: collect(DISTINCT comp.name) AS companies
            ❌ WRONG: comp.name AS company
            ❌ WRONG: comp.name AS organization
            ❌ WRONG: collect(comp.name) AS orgs

            3. Case data MUST use 'case_' prefix for clarity:
            ✅ CORRECT: ca.lr_number AS case_lr_number
            ✅ CORRECT: ca.title AS case_title
            ✅ CORRECT: ca.filing_date AS case_filing_date

            4. ALWAYS use collect(DISTINCT ...) for one-to-many relationships:
            ✅ CORRECT: collect(DISTINCT p.name) AS persons
            ❌ WRONG: p.name AS persons (returns multiple rows)

            5. ORDER BY with aggregation - use the alias, not original variable:
            ✅ CORRECT: ORDER BY case_filing_date DESC
            ❌ WRONG: ORDER BY ca.filing_date DESC (causes syntax error after collect)

            === EXAMPLE QUERIES ===

            Query: "Who was charged in Ponzi schemes?"
            Cypher:
            ```
            MATCH (ca:Case)
            WHERE 'Ponzi Scheme' IN ca.crime_types
            OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(ca)
            OPTIONAL MATCH (comp:Company)-[:INVOLVED_IN]->(ca)
            RETURN 
            ca.lr_number AS case_lr_number,
            ca.title AS case_title,
            ca.filing_date AS case_filing_date,
            collect(DISTINCT p.name) AS persons,
            collect(DISTINCT comp.name) AS companies
            ORDER BY case_filing_date DESC
            LIMIT 20
            ```

            Query: "Show connections for John Smith"
            Cypher:
            ```
            MATCH (p:Person {{name: 'John Smith'}})
            OPTIONAL MATCH (p)-[:WORKED_AT]->(comp:Company)
            OPTIONAL MATCH (p)-[:CHARGED_IN]->(ca:Case)
            RETURN 
            p.name AS person_name,
            collect(DISTINCT comp.name) AS companies,
            collect(DISTINCT ca.lr_number) AS cases
            ```

            Query: "Find recent fraud cases with high penalties"
            Cypher:
            ```
            MATCH (ca:Case)-[:HAS_PENALTY]->(pen:Penalty)
            WHERE 'Fraud' IN ca.crime_types AND pen.amount > 10000000
            OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(ca)
            RETURN 
            ca.lr_number AS case_lr_number,
            ca.title AS case_title,
            ca.filing_date AS case_filing_date,
            sum(pen.amount) AS total_penalty,
            collect(DISTINCT p.name) AS persons
            ORDER BY case_filing_date DESC
            LIMIT 10
            ```

            === ENFORCEMENT ===
            - Use exact aliases: persons, companies, case_lr_number, case_title, case_filing_date
            - Never use synonyms: defendant, accused, organization, corporation
            - Always use collect(DISTINCT ...) for lists
            - After collect(), ORDER BY must use the alias (case_filing_date), not the variable (ca.filing_date)

            Question: {question}

            Cypher query:"""
        
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
        tool_description = """Search the knowledge graph of financial crime cases, people, and companies.

            Use this tool when you need to:
            - Find relationships between people, companies, and cases
            - Identify co-defendants or people who worked at the same companies
            - Find companies involved in multiple cases
            - Get penalty information and amounts for specific entities
            - Discover patterns across cases (repeat offenders, connected crimes)
            - Explore connections not visible in document text

            Do NOT use this tool for:
            - Full case details or document text (use search_sec_documents instead)
            - Current news or recent events (use tavily_search_results_json instead)

            The graph is best for finding CONNECTIONS and RELATIONSHIPS between entities."""
    
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

        