"""Graph query tool for agentic RAG - pattern-based queries
This module is not scalable as it is pattern based. For every
new query, a new method is required"""

import os
from typing import Any, List, Dict, Optional
from loguru import logger
from enum import Enum

from src.kg.neo4j_manager import Neo4jManager

class QueryType(str, Enum):
    """Available query patterns"""
    PONZI_SCHEMES = "ponzi_schemes"
    INSIDER_TRADING = "insider_trading"
    FRAUD_CASES = "fraud_cases"
    PERSON_INFO = "person_info"
    COMPANY_INFO = "company_info"
    HIGH_PENALTIES = "high_penalties"
    RECENT_CASES = "recent_cases"
    CRIME_TYPE = "crime_type"
    PERSON_CONNECTIONS = "person_connections"
    COMPANY_CASES = "company_cases"

class GraphQueryTool:
    """Tool for querying the knowledge graph with predefined patterns"""

    def __init__(self, neo4j_manager: Neo4jManager):
        """Initialize graph query tool
        
        Args:
            neo4j_manager: Neo4jManager instance
        """
        self.manager = neo4j_manager
    
    def query(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """Execute a query based on pattern type
        
        Args:
            query_type: Type of query to execute (from QueryType enum)
            **kwargs: Query-specific parameters
            
        Returns:
            Query results as dictionary
        """
        query_type = query_type.lower()

        filter_params = [
            "crime_type", 
            "min_amount", 
            "max_amount", 
            "after_date",
            "before_date", 
            "person_name", 
            "company_name"
        ]

        if any(param in kwargs for param in filter_params):
            return self.query_with_filters(**kwargs)

        if query_type == QueryType.PONZI_SCHEMES:
            return self.get_ponzi_schemes()

        elif query_type == QueryType.INSIDER_TRADING:
            return self.get_insider_trading_cases()

        elif query_type == QueryType.FRAUD_CASES:
            return self.get_fraud_cases()

        elif query_type == QueryType.PERSON_INFO:
            name = kwargs.get("name")
            if not name:
                return {
                    "error": "Missing required parameter: name",
                }    
            return self.get_person_info(name)

        elif query_type == QueryType.COMPANY_INFO:
            name = kwargs.get("name")
            if not name:
                return {
                     "error": "Missing required parameter: name",
                }
            return self.get_company_info(name)
        
        elif query_type == QueryType.CRIME_TYPE:
            crime_type = kwargs.get("crime_type")
            if not crime_type:
                return {
                    "error": "Missing required parameter: crime_type",
                }
            return self.get_cases_by_crime_type(crime_type)
        
        elif query_type == QueryType.PERSON_CONNECTIONS:
            name = kwargs.get("name")
            if not name:
                return {
                    "error": "Missing required parameter: name",
                }
            return self.get_person_connections(name)
        
        elif query_type == QueryType.COMPANY_CASES:
            name = kwargs.get("name")
            if not name:
                return {"error": "Missing required parameter: name"}
            return self.get_company_cases(name)
        
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    def get_ponzi_schemes(self) -> Dict[str, Any]:
        """Get all Ponzi scheme cases
        
        Returns:
            Dictionary with Ponzi scheme cases
        """
        query = """
        MATCH (c:Case)
        WHERE 'Ponzi Scheme' IN c.crime_types

        OPTIONAL MATCH (p:Person) - [:CHARGED_IN] -> (c)
        OPTIONAL MATCH (comp:Company) - [:INVOLVED_IN] -> (c)
        OPTIONAL MATCH (c) - [:HAS_PENALTY] -> (pen:Penalty)

        RETURN
            c.lr_number AS lr_number,
            c.title AS title,
            c.filing_date AS filing_date,
            c.summary AS summary,
            c.url AS url,
            collect(DISTINCT p.name) AS persons,
            collect(DISTINCT comp.name) AS companies,
            sum(pen.amount) AS total_penalties
        ORDER BY c.filing_date DESC
        LIMIT 50
        """

        results = self.manager.execute_query(query)

        return {
            "query_type": "ponzi_schemes",
            "count": len(results),
            "cases": results,
        }
    
    def get_insider_trading_cases(self) -> Dict[str, Any]:
        """Get all insider trading cases
        
        Returns:
            Dictionary with insider trading cases
        """
        query = """
        MATCH (c:Case)
        WHERE 'Insider Trading' IN c.crime_type

        OPTIONAL MATCH (p:Person) - [:CHARGED_IN] -> (c)
        OPTIONAL MATCH (comp:Company) - [:INVOLED_IN] -> (c)
        OPTIONAL MATCH (c) - [:HAS_PENALTY] -> (pen:Penalty)

        RETURN
            c.lr_number AS lr_number,
            c.title AS title,
            c.filing_date AS filing_date,
            c.summary AS summary,
            c.url AS url,
            collect(DISTINCT p.name) AS persons,
            collect(DISTINCT comp.name) AS companies,
            sum(pen.amount) AS total_penalties
        ORDER BY c.filing_date DESC
        LIMIT 50
        """

        results = self.manager.execute_query(query)

        return {
            "query_type": "insider_trading",
            "count": len(results),
            "cases": results,
        }
    
    def get_fraud_cases(self) -> Dict[str, Any]:
        """Get all fraud cases
        
        Returns:
            Dictionary with fraud cases
        """
        query = """
        MATCH (c:Case)
        WHERE 'Fraud (General)' IN c.crime_types OR 'Securities Fraud' IN c.crime_types
        
        OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(c)
        OPTIONAL MATCH (comp:Company)-[:INVOLVED_IN]->(c)
        
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.filing_date AS filing_date,
            c.summary AS summary,
            c.url AS url,
            collect(DISTINCT p.name) AS persons,
            collect(DISTINCT comp.name) AS companies
        ORDER BY c.filing_date DESC
        LIMIT 50
        """
        
        results = self.manager.execute_query(query)
        
        return {
            "query_type": "fraud_cases",
            "count": len(results),
            "cases": results
        }
    
    def get_person_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a person
        
        Args:
            name: Person's name
            
        Returns:
            Dictionary with person information
        """
        query = """
        MATCH (p:Person)
        WHERE toLower(p.name) CONTAINS toLower($name)
        
        OPTIONAL MATCH (p)-[:WORKED_AT]->(c:Company)
        OPTIONAL MATCH (p)-[:CHARGED_IN]->(case:Case)
        OPTIONAL MATCH (p)-[:RECEIVED_PENALTY]->(pen:Penalty)
        
        RETURN 
            p.name AS name,
            p.role AS role,
            collect(DISTINCT c.name) AS companies,
            collect(DISTINCT {
                lr_number: case.lr_number,
                title: case.title,
                crime_types: case.crime_types,
                filing_date: case.filing_date
            }) AS cases,
            collect(DISTINCT {
                type: pen.penalty_type,
                amount: pen.amount,
                description: pen.description
            }) AS penalties
        LIMIT 10
        """
        
        results = self.manager.execute_query(query, {"name": name})
        
        if not results:
            return {
                "query_type": "person_info",
                "found": False,
                "message": f"No person found matching '{name}'"
            }
        
        return {
            "query_type": "person_info",
            "found": True,
            "count": len(results),
            "persons": results
        }
    
    def get_company_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a company
        
        Args:
            name: Company name
            
        Returns:
            Dictionary with company information
        """
        query = """
        MATCH (c:Company)
        WHERE toLower(c.name) CONTAINS toLower($name)
        
        OPTIONAL MATCH (c)-[:INVOLVED_IN]->(case:Case)
        OPTIONAL MATCH (p:Person)-[:WORKED_AT]->(c)
        OPTIONAL MATCH (c)-[:RECEIVED_PENALTY]->(pen:Penalty)
        
        RETURN 
            c.name AS name,
            c.ticker AS ticker,
            c.industry AS industry,
            c.description AS description,
            collect(DISTINCT {
                lr_number: case.lr_number,
                title: case.title,
                crime_types: case.crime_types
            }) AS cases,
            collect(DISTINCT p.name) AS employees,
            collect(DISTINCT {
                type: pen.penalty_type,
                amount: pen.amount
            }) AS penalties
        LIMIT 10
        """
        
        results = self.manager.execute_query(query, {"name": name})
        
        if not results:
            return {
                "query_type": "company_info",
                "found": False,
                "message": f"No company found matching '{name}'"
            }
        
        return {
            "query_type": "company_info",
            "found": True,
            "count": len(results),
            "companies": results
        }

    def get_high_penalty_cases(self, min_amount: float = 10000000) -> Dict[str, Any]:
        """Get cases with penalties above a threshold
        
        Args:
            min_amount: Minimum penalty amount (default: $10M)
            
        Returns:
            Dictionary with high-penalty cases
        """
        query = """
        MATCH (c:Case)-[:HAS_PENALTY]->(p:Penalty)
        WHERE p.amount >= $min_amount
        WITH c, sum(p.amount) AS total_penalty
        ORDER BY total_penalty DESC
        LIMIT 20
        
        OPTIONAL MATCH (person:Person)-[:CHARGED_IN]->(c)
        OPTIONAL MATCH (company:Company)-[:INVOLVED_IN]->(c)
        
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.crime_types AS crime_types,
            c.filing_date AS filing_date,
            c.url AS url,
            total_penalty,
            collect(DISTINCT person.name) AS persons,
            collect(DISTINCT company.name) AS companies
        """
        
        results = self.manager.execute_query(query, {"min_amount": min_amount})
        
        return {
            "query_type": "high_penalties",
            "min_amount": min_amount,
            "count": len(results),
            "cases": results
        }
    
    def get_recent_cases(self, limit: int = 10) -> Dict[str, Any]:
        """Get most recent cases
        
        Args:
            limit: Number of cases to return
            
        Returns:
            Dictionary with recent cases
        """
        query = """
        MATCH (c:Case)
        
        OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(c)
        OPTIONAL MATCH (comp:Company)-[:INVOLVED_IN]->(c)
        OPTIONAL MATCH (c)-[:HAS_PENALTY]->(pen:Penalty)
        
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.crime_types AS crime_types,
            c.filing_date AS filing_date,
            c.summary AS summary,
            c.url AS url,
            collect(DISTINCT p.name) AS persons,
            collect(DISTINCT comp.name) AS companies,
            sum(pen.amount) AS total_penalties
        ORDER BY c.filing_date DESC
        LIMIT $limit
        """
        
        results = self.manager.execute_query(query, {"limit": limit})
        
        return {
            "query_type": "recent_cases",
            "count": len(results),
            "cases": results
        }

    def get_cases_by_crime_type(self, crime_type: str) -> Dict[str, Any]:
        """Get cases by crime type
        
        Args:
            crime_type: Crime type to search for
            
        Returns:
            Dictionary with matching cases
        """
        query = """
        MATCH (c:Case)
        WHERE ANY(crime IN c.crime_types WHERE toLower(crime) CONTAINS toLower($crime_type))
        
        OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(c)
        OPTIONAL MATCH (comp:Company)-[:INVOLVED_IN]->(c)
        OPTIONAL MATCH (c)-[:HAS_PENALTY]->(pen:Penalty)
        
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.crime_types AS crime_types,
            c.filing_date AS filing_date,
            c.summary AS summary,
            c.url AS url,
            collect(DISTINCT p.name) AS persons,
            collect(DISTINCT comp.name) AS companies,
            sum(pen.amount) AS total_penalties
        ORDER BY c.filing_date DESC
        LIMIT 50
        """
        
        results = self.manager.execute_query(query, {"crime_type": crime_type})
        
        return {
            "query_type": "crime_type",
            "crime_type": crime_type,
            "count": len(results),
            "cases": results
        }

    def get_person_connections(self, name: str) -> Dict[str, Any]:
        """Get all connections for a person (co-defendants, shared companies)
        
        Args:
            name: Person's name
            
        Returns:
            Dictionary with person connections
        """
        query = """
        MATCH (p:Person)
        WHERE toLower(p.name) CONTAINS toLower($name)
        
        // Find co-defendants (people charged in same cases)
        OPTIONAL MATCH (p)-[:CHARGED_IN]->(case:Case)<-[:CHARGED_IN]-(codefendant:Person)
        WHERE p <> codefendant
        
        // Find colleagues (people who worked at same companies)
        OPTIONAL MATCH (p)-[:WORKED_AT]->(company:Company)<-[:WORKED_AT]-(colleague:Person)
        WHERE p <> colleague
        
        RETURN 
            p.name AS name,
            p.role AS role,
            collect(DISTINCT {
                name: codefendant.name,
                relationship: 'co-defendant',
                case: case.lr_number
            }) AS codefendants,
            collect(DISTINCT {
                name: colleague.name,
                relationship: 'colleague',
                company: company.name
            }) AS colleagues
        LIMIT 5
        """
        
        results = self.manager.execute_query(query, {"name": name})
        
        if not results:
            return {
                "query_type": "person_connections",
                "found": False,
                "message": f"No person found matching '{name}'"
            }
        
        return {
            "query_type": "person_connections",
            "found": True,
            "count": len(results),
            "connections": results
        }

    def get_company_cases(self, name: str) -> Dict[str, Any]:
        """Get all cases involving a company
        
        Args:
            name: Company name
            
        Returns:
            Dictionary with company cases
        """
        query = """
        MATCH (c:Company)
        WHERE toLower(c.name) CONTAINS toLower($name)
        
        MATCH (c)-[:INVOLVED_IN]->(case:Case)
        
        OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(case)
        OPTIONAL MATCH (case)-[:HAS_PENALTY]->(pen:Penalty)
        
        RETURN 
            c.name AS company_name,
            c.ticker AS ticker,
            collect(DISTINCT {
                lr_number: case.lr_number,
                title: case.title,
                crime_types: case.crime_types,
                filing_date: case.filing_date,
                persons: collect(DISTINCT p.name),
                total_penalty: sum(pen.amount)
            }) AS cases
        LIMIT 5
        """
        
        results = self.manager.execute_query(query, {"name": name})
        
        if not results:
            return {
                "query_type": "company_cases",
                "found": False,
                "message": f"No company found matching '{name}'"
            }
        
        return {
            "query_type": "company_cases",
            "found": True,
            "count": len(results),
            "companies": results
        }
    
    def query_with_filters(
        self,
        crime_type: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        person_name: Optional[str] = None,
        company_name: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Query cases with flexible filters - matches existing RAG API
        
        Args:
            crime_type: Filter by crime type (partial match)
            min_amount: Minimum penalty amount
            max_amount: Maximum penalty amount
            after_date: Cases filed after this date (YYYY-MM-DD)
            before_date: Cases filed before this date (YYYY-MM-DD)
            person_name: Filter by person name (partial match)
            company_name: Filter by company name (partial match)
            limit: Maximum number of results
            
        Returns:
            Dictionary with filtered cases
        """
        # Build WHERE clauses dynamically
        where_clauses = []
        params = {"limit": limit}
        
        # Crime type filter
        if crime_type:
            where_clauses.append("ANY(crime IN c.crime_types WHERE toLower(crime) CONTAINS toLower($crime_type))")
            params["crime_type"] = crime_type
        
        # Date filters
        if after_date:
            where_clauses.append("c.filing_date >= date($after_date)")
            params["after_date"] = after_date
        
        if before_date:
            where_clauses.append("c.filing_date <= date($before_date)")
            params["before_date"] = before_date
        
        # Build base query
        query_parts = ["MATCH (c:Case)"]
        
        # Optional person filter
        if person_name:
            query_parts.append("MATCH (p:Person)-[:CHARGED_IN]->(c)")
            where_clauses.append("toLower(p.name) CONTAINS toLower($person_name)")
            params["person_name"] = person_name
        else:
            query_parts.append("OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(c)")
        
        # Optional company filter
        if company_name:
            query_parts.append("MATCH (comp:Company)-[:INVOLVED_IN]->(c)")
            where_clauses.append("toLower(comp.name) CONTAINS toLower($company_name)")
            params["company_name"] = company_name
        else:
            query_parts.append("OPTIONAL MATCH (comp:Company)-[:INVOLVED_IN]->(c)")
        
        # Penalties (for amount filtering)
        query_parts.append("OPTIONAL MATCH (c)-[:HAS_PENALTY]->(pen:Penalty)")
        
        # Add WHERE clause if filters exist
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Aggregate penalties for amount filtering
        query_parts.append("""
        WITH c, 
             collect(DISTINCT p.name) AS persons,
             collect(DISTINCT comp.name) AS companies,
             sum(pen.amount) AS total_penalties
        """)
        
        # Amount filters (after aggregation)
        amount_filters = []
        if min_amount is not None:
            amount_filters.append("total_penalties >= $min_amount")
            params["min_amount"] = min_amount
        
        if max_amount is not None:
            amount_filters.append("total_penalties <= $max_amount")
            params["max_amount"] = max_amount
        
        if amount_filters:
            query_parts.append("WHERE " + " AND ".join(amount_filters))
        
        # Return results
        query_parts.append("""
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.crime_types AS crime_types,
            c.filing_date AS filing_date,
            c.summary AS summary,
            c.url AS url,
            persons,
            companies,
            total_penalties
        ORDER BY c.filing_date DESC
        LIMIT $limit
        """)
        
        query = "\n".join(query_parts)
        
        results = self.manager.execute_query(query, params)
        
        # Build filter description
        filters_applied = []
        if crime_type:
            filters_applied.append(f"crime_type='{crime_type}'")
        if min_amount:
            filters_applied.append(f"min_amount=${min_amount:,.0f}")
        if max_amount:
            filters_applied.append(f"max_amount=${max_amount:,.0f}")
        if after_date:
            filters_applied.append(f"after={after_date}")
        if before_date:
            filters_applied.append(f"before={before_date}")
        if person_name:
            filters_applied.append(f"person='{person_name}'")
        if company_name:
            filters_applied.append(f"company='{company_name}'")
        
        return {
            "query_type": "filtered_cases",
            "filters": ", ".join(filters_applied) if filters_applied else "none",
            "count": len(results),
            "cases": results
        }

    def format_results_for_llm(self, results: Dict[str, Any]) -> str:
        """Format query results as text for LLM context
        
        Args:
            results: Query results dictionary
            
        Returns:
            Formatted string for LLM
        """
        query_type = results.get("query_type", "unknown")
        
        if "error" in results:
            return f"Error: {results['error']}"
        
        if not results.get("found", True):
            return results.get("message", "No results found")
        
        # Format based on query type
        query_types = [
            "ponzi_schemes", 
            "insider_trading", 
            "fraud_cases", 
            "crime_type", 
            "recent_cases", 
            "high_penalties"
        ]
        if query_type in query_types:
            cases = results.get("cases", [])
            if not cases:
                return "No cases found."
            
            output = [f"Found {results['count']} cases:\n"]
            for i, case in enumerate(cases[:10], 1):  # Limit to 10 for context
                output.append(f"{i}. {case['title']} ({case['lr_number']})")
                output.append(f"   Crime Types: {', '.join(case.get('crime_types', [query_type]))}")
                output.append(f"   Filing Date: {case.get('filing_date', 'N/A')}")
                if case.get('persons'):
                    output.append(f"   Persons: {', '.join(case['persons'][:5])}")
                if case.get('total_penalties'):
                    output.append(f"   Total Penalties: ${case['total_penalties']:,.0f}")
                output.append(f"   URL: {case.get('url', 'N/A')}\n")
            
            return "\n".join(output)
        
        elif query_type == "person_info":
            persons = results.get("persons", [])
            if not persons:
                return "No persons found."
            
            output = []
            for person in persons:
                output.append(f"Name: {person['name']}")
                if person.get('role'):
                    output.append(f"Role: {person['role']}")
                if person.get('companies'):
                    output.append(f"Companies: {', '.join(person['companies'])}")
                if person.get('cases'):
                    output.append(f"Cases: {len(person['cases'])}")
                    for case in person['cases'][:3]:
                        if case.get('lr_number'):
                            output.append(f"  - {case['lr_number']}: {case['title']}")
                output.append("")
            
            return "\n".join(output)
        
        elif query_type == "company_info":
            companies = results.get("companies", [])
            if not companies:
                return "No companies found."
            
            output = []
            for company in companies:
                output.append(f"Company: {company['name']}")
                if company.get('ticker'):
                    output.append(f"Ticker: {company['ticker']}")
                if company.get('industry'):
                    output.append(f"Industry: {company['industry']}")
                if company.get('cases'):
                    output.append(f"Cases: {len(company['cases'])}")
                output.append("")
            
            return "\n".join(output)
        
        else:
            return str(results)
    

if __name__ == "__main__":

    with Neo4jManager() as manager:
        tool = GraphQueryTool(manager)
        
        print("=" * 70)
        print("GRAPH QUERY TOOL EXAMPLES")
        print("=" * 70)
        
        # Example 1: Get Ponzi schemes
        print("\n1. Ponzi Schemes:")
        results = tool.query("ponzi_schemes")
        print(tool.format_results_for_llm(results))
        
        # Example 2: Get person info
        print("\n2. Person Info:")
        results = tool.query("person_info", name="Wood")
        print(tool.format_results_for_llm(results))
        
        # Example 3: High penalty cases
        print("\n3. High Penalty Cases (>$10M):")
        results = tool.query("high_penalties", min_amount=10000000)
        print(tool.format_results_for_llm(results))
        
        # Example 4: Filter by multiple criteria
        print("\n4. Filtered Query (Ponzi + Amount + Date):")
        results = tool.query_with_filters(
            crime_type="Ponzi",
            min_amount=5000000,
            after_date="2023-01-01",
            limit=10
        )
        print(f"Found {results['count']} cases with filters: {results['filters']}")
        print(tool.format_results_for_llm(results))

