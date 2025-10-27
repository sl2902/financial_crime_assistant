"""Example queries for exploring the financial crime knowledge graph"""

from typing import List, Dict, Any
from loguru import logger
from tabulate import tabulate

from src.kg.neo4j_manager import Neo4jManager


class GraphQueryExamples:
    """Collection of useful graph queries."""
    
    def __init__(self, manager: Neo4jManager):
        """Initialize with Neo4j manager.
        
        Args:
            manager: Neo4jManager instance
        """
        self.manager = manager
    
    def get_all_ponzi_schemes(self) -> List[Dict[str, Any]]:
        """Get all Ponzi scheme cases with details.
        
        Returns:
            List of Ponzi scheme cases with persons and companies involved
        """
        query = """
        MATCH (c:Case)
        WHERE 'Ponzi Scheme' IN c.crime_types
        
        OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(c)
        OPTIONAL MATCH (comp:Company)-[:INVOLVED_IN]->(c)
        OPTIONAL MATCH (c)-[:HAS_PENALTY]->(pen:Penalty)
        
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.filing_date AS filing_date,
            c.url AS url,
            collect(DISTINCT p.name) AS persons,
            collect(DISTINCT comp.name) AS companies,
            sum(pen.amount) AS total_penalties
        ORDER BY c.filing_date DESC
        """
        
        results = self.manager.execute_query(query)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PONZI SCHEMES FOUND: {len(results)}")
        logger.info(f"{'='*80}\n")
        
        # Format for display
        table_data = []
        for i, result in enumerate(results, 1):
            table_data.append([
                i,
                result['lr_number'],
                result['title'][:50] + '...' if len(result['title']) > 50 else result['title'],
                result['filing_date'],
                len(result['persons']),
                len(result['companies']),
                f"${result['total_penalties']:,.0f}" if result['total_penalties'] else "N/A"
            ])
        
        headers = ["#", "LR Number", "Title", "Filing Date", "Persons", "Companies", "Total Penalties"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return results
    
    def get_ponzi_scheme_details(self, lr_number: str) -> Dict[str, Any]:
        """Get detailed information about a specific Ponzi scheme.
        
        Args:
            lr_number: Case LR number
            
        Returns:
            Detailed case information
        """
        query = """
        MATCH (c:Case {lr_number: $lr_number})
        
        OPTIONAL MATCH (p:Person)-[:CHARGED_IN]->(c)
        OPTIONAL MATCH (p)-[:WORKED_AT]->(comp:Company)
        OPTIONAL MATCH (pen:Penalty)<-[:RECEIVED_PENALTY]-(p)
        
        OPTIONAL MATCH (company:Company)-[:INVOLVED_IN]->(c)
        OPTIONAL MATCH (company_pen:Penalty)<-[:RECEIVED_PENALTY]-(company)
        
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.crime_types AS crime_types,
            c.filing_date AS filing_date,
            c.summary AS summary,
            c.url AS url,
            collect(DISTINCT {
                name: p.name, 
                role: p.role,
                companies: collect(DISTINCT comp.name),
                penalties: collect(DISTINCT {
                    type: pen.penalty_type,
                    amount: pen.amount,
                    description: pen.description
                })
            }) AS persons,
            collect(DISTINCT {
                name: company.name,
                ticker: company.ticker,
                industry: company.industry,
                penalties: collect(DISTINCT {
                    type: company_pen.penalty_type,
                    amount: company_pen.amount,
                    description: company_pen.description
                })
            }) AS companies
        """
        
        results = self.manager.execute_query(query, {"lr_number": lr_number})
        
        if not results:
            logger.warning(f"No case found with LR number: {lr_number}")
            return {}
        
        case = results[0]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"CASE DETAILS: {case['lr_number']}")
        logger.info(f"{'='*80}")
        logger.info(f"Title: {case['title']}")
        logger.info(f"Crime Types: {', '.join(case['crime_types'])}")
        logger.info(f"Filing Date: {case['filing_date']}")
        logger.info(f"URL: {case['url']}")
        logger.info(f"\nSummary:\n{case['summary']}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PERSONS CHARGED ({len(case['persons'])})")
        logger.info(f"{'='*80}")
        for person in case['persons']:
            if person['name']:  # Skip empty entries
                logger.info(f"\n  👤 {person['name']}")
                if person['role']:
                    logger.info(f"     Role: {person['role']}")
                if person['companies']:
                    logger.info(f"     Companies: {', '.join(person['companies'])}")
                if person['penalties']:
                    logger.info(f"     Penalties:")
                    for penalty in person['penalties']:
                        if penalty['type']:
                            amount_str = f"${penalty['amount']:,.0f}" if penalty['amount'] else "N/A"
                            logger.info(f"       - {penalty['type']}: {amount_str}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPANIES INVOLVED ({len(case['companies'])})")
        logger.info(f"{'='*80}")
        for company in case['companies']:
            if company['name']:  # Skip empty entries
                logger.info(f"\n  🏢 {company['name']}")
                if company['ticker']:
                    logger.info(f"     Ticker: {company['ticker']}")
                if company['industry']:
                    logger.info(f"     Industry: {company['industry']}")
                if company['penalties']:
                    logger.info(f"     Penalties:")
                    for penalty in company['penalties']:
                        if penalty['type']:
                            amount_str = f"${penalty['amount']:,.0f}" if penalty['amount'] else "N/A"
                            logger.info(f"       - {penalty['type']}: {amount_str}")
        
        return case
    
    def get_top_penalties(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cases with highest total penalties.
        
        Args:
            limit: Number of results to return
            
        Returns:
            Cases with highest penalties
        """
        query = """
        MATCH (c:Case)-[:HAS_PENALTY]->(p:Penalty)
        WHERE p.amount IS NOT NULL
        WITH c, sum(p.amount) AS total_penalty
        ORDER BY total_penalty DESC
        LIMIT $limit
        
        OPTIONAL MATCH (person:Person)-[:CHARGED_IN]->(c)
        OPTIONAL MATCH (company:Company)-[:INVOLVED_IN]->(c)
        
        RETURN 
            c.lr_number AS lr_number,
            c.title AS title,
            c.crime_types AS crime_types,
            total_penalty,
            collect(DISTINCT person.name) AS persons,
            collect(DISTINCT company.name) AS companies
        """
        
        results = self.manager.execute_query(query, {"limit": limit})
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP {limit} CASES BY TOTAL PENALTIES")
        logger.info(f"{'='*80}\n")
        
        table_data = []
        for i, result in enumerate(results, 1):
            table_data.append([
                i,
                result['lr_number'],
                result['title'][:40] + '...' if len(result['title']) > 40 else result['title'],
                ', '.join(result['crime_types']),
                f"${result['total_penalty']:,.0f}",
                len(result['persons']),
                len(result['companies'])
            ])
        
        headers = ["Rank", "LR Number", "Title", "Crime Types", "Total Penalties", "Persons", "Companies"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return results
    
    def find_person_connections(self, person_name: str) -> Dict[str, Any]:
        """Find all connections for a person (companies, cases, penalties).
        
        Args:
            person_name: Person's name
            
        Returns:
            Person's connections
        """
        query = """
        MATCH (p:Person {name: $person_name})
        
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
                crime_types: case.crime_types
            }) AS cases,
            collect(DISTINCT {
                type: pen.penalty_type,
                amount: pen.amount,
                description: pen.description
            }) AS penalties
        """
        
        results = self.manager.execute_query(query, {"person_name": person_name})
        
        if not results:
            logger.warning(f"Person not found: {person_name}")
            return {}
        
        person = results[0]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PERSON: {person['name']}")
        logger.info(f"{'='*80}")
        logger.info(f"Role: {person['role'] or 'N/A'}")
        logger.info(f"\nCompanies ({len(person['companies'])}):")
        for company in person['companies']:
            logger.info(f"  🏢 {company}")
        
        logger.info(f"\nCases ({len(person['cases'])}):")
        for case in person['cases']:
            if case['lr_number']:
                logger.info(f"  ⚖️  {case['lr_number']}: {case['title']}")
                logger.info(f"      Crime Types: {', '.join(case['crime_types'])}")
        
        logger.info(f"\nPenalties ({len(person['penalties'])}):")
        total = 0
        for penalty in person['penalties']:
            if penalty['type']:
                amount_str = f"${penalty['amount']:,.0f}" if penalty['amount'] else "N/A"
                logger.info(f"  💰 {penalty['type']}: {amount_str}")
                if penalty['amount']:
                    total += penalty['amount']
        
        if total > 0:
            logger.info(f"\n  Total Monetary Penalties: ${total:,.0f}")
        
        return person
    
    def get_companies_with_multiple_cases(self) -> List[Dict[str, Any]]:
        """Find companies involved in multiple cases.
        
        Returns:
            Companies with multiple cases
        """
        query = """
        MATCH (c:Company)-[:INVOLVED_IN]->(case:Case)
        WITH c, count(case) AS case_count, collect(case.lr_number) AS cases
        WHERE case_count > 1
        ORDER BY case_count DESC
        
        RETURN 
            c.name AS company,
            c.ticker AS ticker,
            case_count,
            cases
        """
        
        results = self.manager.execute_query(query)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPANIES WITH MULTIPLE CASES: {len(results)}")
        logger.info(f"{'='*80}\n")
        
        for result in results:
            logger.info(f"🏢 {result['company']} ({result['ticker'] or 'N/A'})")
            logger.info(f"   Cases: {result['case_count']}")
            logger.info(f"   LR Numbers: {', '.join(result['cases'])}\n")
        
        return results


# Command-line interface
if __name__ == "__main__":
    import sys
    
    with Neo4jManager() as manager:
        queries = GraphQueryExamples(manager)
        
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "ponzi":
                queries.get_all_ponzi_schemes()
            
            elif command == "details" and len(sys.argv) > 2:
                lr_number = sys.argv[2]
                queries.get_ponzi_scheme_details(lr_number)
            
            elif command == "top":
                limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                queries.get_top_penalties(limit)
            
            elif command == "person" and len(sys.argv) > 2:
                person_name = " ".join(sys.argv[2:])
                queries.find_person_connections(person_name)
            
            elif command == "companies":
                queries.get_companies_with_multiple_cases()
            
            else:
                print("Usage:")
                print("  python query_examples.py ponzi")
                print("  python query_examples.py details LR-XXXXX")
                print("  python query_examples.py top [limit]")
                print("  python query_examples.py person <name>")
                print("  python query_examples.py companies")
        
        else:
            # Run all example queries
            print("\n Running example queries...\n")
            
            queries.get_all_ponzi_schemes()
            input("\nPress Enter to continue to top penalties...")
            
            queries.get_top_penalties(10)
            input("\nPress Enter to continue to companies with multiple cases...")
            
            queries.get_companies_with_multiple_cases()