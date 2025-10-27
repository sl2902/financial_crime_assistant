"""Graph builder to load extracted entities into Neo4j knowledge graph"""

import os
import sys
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
from loguru import logger

from src.schemas.entity_schemas import ExtractedEntities
from src.kg.neo4j_manager import Neo4jManager

class GraphBuilder:
    """Builds knowledge graph from extracted entities"""

    def __init__(self, neo4j_manager: Neo4jManager):
        """Initialize graph builder
        
        Args:
            neo4j_manager: Neo4jManager instance for database operations
        """
        self.manager = neo4j_manager
    
    def create_person_node(self, person_data: Dict) -> Dict:
        """Create or merge a Person node
        
        Args:
            person_data: Dictionary with person information
            
        Returns:
            Query execution statistics
        """
        query = """
        MERGE (p:Person {name: $name})
        ON CREATE SET
            p.role = $role,
            p.created_at = datetime()
        ON MATCH SET
            p.role = COALESCE($role, p.role),
            p.updated_at = datetime()
        RETURN p
        """

        params = {
            "name": person_data["name"],
            "role": person_data["role"],
        }

        return self.manager.execute_write(query, params)
    
    def create_company_node(self, company_data: Dict) -> Dict:
        """Create or merge a Company node
        
        Args:
            company_data: Dictionary with company information
            
        Returns:
            Query execution statistics
        """
        query = """
        MERGE (c:Company {name: $name})
        ON CREATE SET 
            c.ticker = $ticker,
            c.industry = $industry,
            c.description = $description,
            c.created_at = datetime()
        ON MATCH SET
            c.ticker = COALESCE($ticker, c.ticker),
            c.industry = COALESCE($industry, c.industry),
            c.description = COALESCE($description, c.description),
            c.updated_at = datetime()
        RETURN c
        """

        params = {
            "name": company_data["name"],
            "ticker": company_data.get("ticker"),
            "industry": company_data.get("industry"),
            "description": company_data.get("description"),
        }
        
        return self.manager.execute_write(query, params)
    
    def create_case_node(self, case_data: Dict) -> Dict:
        """Create or merge a Case node.
        
        Args:
            case_data: Dictionary with case information
            
        Returns:
            Query execution statistics
        """
        query = """
        MERGE (c:Case {lr_number: $lr_number})
        ON CREATE SET 
            c.title = $title,
            c.crime_types = $crime_types,
            c.filing_date = date($filing_date),
            c.summary = $summary,
            c.url = $url,
            c.created_at = datetime()
        ON MATCH SET
            c.title = $title,
            c.crime_types = $crime_types,
            c.filing_date = date($filing_date),
            c.summary = $summary,
            c.url = $url,
            c.updated_at = datetime()
        RETURN c
        """
        
        params = {
            "lr_number": case_data["lr_number"],
            "title": case_data["title"],
            "crime_types": case_data["crime_types"],
            "filing_date": str(case_data["filing_date"]),
            "summary": case_data.get("summary"),
            "url": case_data["url"],
        }

        return self.manager.execute_write(query, params)

    
    def create_penalty_node(self, penalty_data: Dict, lr_number: str) -> str:
        """Create a Penalty node with unique ID.
        
        Args:
            penalty_data: Dictionary with penalty information
            lr_number: Associated case LR number
            
        Returns:
            Penalty node ID
        """
        # Create unique ID for penalty (since penalties aren't unique by themselves)
        penalty_id = f"{lr_number}-{penalty_data['penalty_type']}-{penalty_data.get('recipient', 'unknown')}"
        
        query = """
        CREATE (p:Penalty {
            id: $id,
            penalty_type: $penalty_type,
            amount: $amount,
            description: $description,
            recipient: $recipient,
            created_at: datetime()
        })
        RETURN p
        """
        
        params = {
            "id": penalty_id,
            "penalty_type": penalty_data["penalty_type"],
            "amount": penalty_data.get("amount"),
            "description": penalty_data["description"],
            "recipient": penalty_data.get("recipient"),
        }
        
        self.manager.execute_write(query, params)
        return penalty_id
    
    def create_charged_in_relationship(self, person_name: str, lr_number: str):
        """Create CHARGED_IN relationship between Person and Case
        
        Args:
            person_name: Person's name
            lr_number: Case LR number
        """
        query = """
        MATCH (p:Person {name: $person_name})
        MATCH (c:Case {lr_number: $lr_number})
        MERGE (p)-[:CHARGED_IN]->(c)
        """
        
        self.manager.execute_write(query, {
            "person_name": person_name,
            "lr_number": lr_number
        })
    
    def create_worked_at_relationship(self, person_name: str, company_name: str):
        """Create WORKED_AT relationship between Person and Company
        
        Args:
            person_name: Person's name
            company_name: Company name
        """
        query = """
        MATCH (p:Person {name: $person_name})
        MATCH (c:Company {name: $company_name})
        MERGE (p)-[:WORKED_AT]->(c)
        """
        
        self.manager.execute_write(query, {
            "person_name": person_name,
            "company_name": company_name
        })
    
    def create_involved_in_relationship(self, company_name: str, lr_number: str):
        """Create INVOLVED_IN relationship between Company and Case
        
        Args:
            company_name: Company name
            lr_number: Case LR number
        """
        query = """
        MATCH (c:Company {name: $company_name})
        MATCH (ca:Case {lr_number: $lr_number})
        MERGE (c)-[:INVOLVED_IN]->(ca)
        """
        
        self.manager.execute_write(query, {
            "company_name": company_name,
            "lr_number": lr_number
        })
    
    def create_received_penalty_relationship(self, recipient_name: str, penalty_id: str):
        """Create RECEIVED_PENALTY relationship between Person/Company and Penalty
        
        Args:
            recipient_name: Person or Company name
            penalty_id: Penalty node ID
        """
        # Try to match as Person first, then Company
        query = """
        MATCH (p:Penalty {id: $penalty_id})
        OPTIONAL MATCH (person:Person {name: $recipient_name})
        OPTIONAL MATCH (company:Company {name: $recipient_name})
        
        FOREACH (_ IN CASE WHEN person IS NOT NULL THEN [1] ELSE [] END |
            MERGE (person)-[:RECEIVED_PENALTY]->(p)
        )
        
        FOREACH (_ IN CASE WHEN company IS NOT NULL THEN [1] ELSE [] END |
            MERGE (company)-[:RECEIVED_PENALTY]->(p)
        )
        """
        
        self.manager.execute_write(query, {
            "recipient_name": recipient_name,
            "penalty_id": penalty_id
        })
    
    def create_has_penalty_relationship(self, lr_number: str, penalty_id: str):
        """Create HAS_PENALTY relationship between Case and Penalty
        
        Args:
            lr_number: Case LR number
            penalty_id: Penalty node ID
        """
        query = """
        MATCH (c:Case {lr_number: $lr_number})
        MATCH (p:Penalty {id: $penalty_id})
        MERGE (c)-[:HAS_PENALTY]->(p)
        """
        
        self.manager.execute_write(query, {
            "lr_number": lr_number,
            "penalty_id": penalty_id
        })
    
    def load_entity(self, entity: ExtractedEntities) -> Dict:
        """Load a single ExtractedEntities object into the graph
        
        Args:
            entity: ExtractedEntities object
            
        Returns:
            Statistics about nodes and relationships created
        """
        stats = {
            "persons": 0,
            "companies": 0,
            "cases": 0,
            "penalties": 0,
            "relationships": 0,
        }
        
        lr_number = entity.lr_number
        
        # 1. Create Case node
        self.create_case_node({
            "lr_number": entity.case.lr_number,
            "title": entity.case.title,
            "crime_types": [ct.value for ct in entity.case.crime_types],
            "filing_date": entity.case.filing_date,
            "summary": entity.case.summary,
            "url": entity.case.url,
        })
        stats["cases"] += 1
        
        # 2. Create Person nodes and relationships
        for person in entity.persons:
            self.create_person_node({
                "name": person.name,
                "role": person.role,
            })
            stats["persons"] += 1
            
            # Person -> Case
            self.create_charged_in_relationship(person.name, lr_number)
            stats["relationships"] += 1
            
            # Person -> Company (for each affiliation)
            for company_name in person.company_affiliations:
                # Ensure company exists
                self.create_company_node({"name": company_name})
                self.create_worked_at_relationship(person.name, company_name)
                stats["relationships"] += 1
        
        # 3. Create Company nodes and relationships
        for company in entity.companies:
            self.create_company_node({
                "name": company.name,
                "ticker": company.ticker,
                "industry": company.industry,
                "description": company.description,
            })
            stats["companies"] += 1
            
            # Company -> Case
            self.create_involved_in_relationship(company.name, lr_number)
            stats["relationships"] += 1
        
        # 4. Create Penalty nodes and relationships
        for penalty in entity.penalties:
            penalty_id = self.create_penalty_node({
                "penalty_type": penalty.penalty_type.value,
                "amount": penalty.amount,
                "description": penalty.description,
                "recipient": penalty.recipient,
            }, lr_number)
            stats["penalties"] += 1
            
            # Case -> Penalty
            self.create_has_penalty_relationship(lr_number, penalty_id)
            stats["relationships"] += 1
            
            # Person/Company -> Penalty (if recipient specified)
            if penalty.recipient:
                self.create_received_penalty_relationship(penalty.recipient, penalty_id)
                stats["relationships"] += 1
        
        return stats
    
    def load_from_jsonl(self, file_path: str) -> Dict:
        """Load entities from a JSONL file into the graph
        
        Args:
            file_path: Path to JSONL file with extracted entities
            
        Returns:
            Overall statistics
        """
        total_stats = {
            "persons": 0,
            "companies": 0,
            "cases": 0,
            "penalties": 0,
            "relationships": 0,
            "documents_processed": 0,
            "errors": 0,
        }
        
        logger.info(f"Loading entities from {file_path}")
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    entity = ExtractedEntities(**data)
                    
                    stats = self.load_entity(entity)
                    
                    # Accumulate stats
                    for key in ["persons", "companies", "cases", "penalties", "relationships"]:
                        total_stats[key] += stats[key]
                    
                    total_stats["documents_processed"] += 1
                    
                    if line_num % 10 == 0:
                        logger.info(f"  Processed {line_num} documents...")
                
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    total_stats["errors"] += 1
                    continue
        
        logger.info(f"   Completed loading {file_path}")
        logger.info(f"   Documents: {total_stats['documents_processed']}")
        logger.info(f"   Persons: {total_stats['persons']}")
        logger.info(f"   Companies: {total_stats['companies']}")
        logger.info(f"   Cases: {total_stats['cases']}")
        logger.info(f"   Penalties: {total_stats['penalties']}")
        logger.info(f"   Relationships: {total_stats['relationships']}")
        logger.info(f"   Errors: {total_stats['errors']}")
        
        return total_stats

    
    def load_from_directory(self, directory_path: str, pattern: str = "*.jsonl") -> Dict:
        """Load entities from all JSONL files in a directory
        
        Args:
            directory_path: Path to directory containing JSONL files
            pattern: File pattern to match (default: *.jsonl)
            
        Returns:
            Overall statistics across all files
        """
        directory = Path(directory_path)
        files = sorted(directory.glob(pattern))
        
        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory_path}")
            return {}
        
        logger.info(f"Found {len(files)} files to process")
        
        total_stats = {
            "persons": 0,
            "companies": 0,
            "cases": 0,
            "penalties": 0,
            "relationships": 0,
            "documents_processed": 0,
            "errors": 0,
            "files_processed": 0,
        }
        
        for file_path in files:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing file: {file_path.name}")
            logger.info(f"{'='*70}")
            
            stats = self.load_from_jsonl(str(file_path))
            
            # Accumulate stats
            for key in stats:
                total_stats[key] += stats[key]
            
            total_stats["files_processed"] += 1
        
        logger.info(f"\n{'='*70}")
        logger.info("FINAL STATISTICS")
        logger.info(f"{'='*70}")
        logger.info(f"Files processed: {total_stats['files_processed']}")
        logger.info(f"Documents processed: {total_stats['documents_processed']}")
        logger.info(f"Total persons: {total_stats['persons']}")
        logger.info(f"Total companies: {total_stats['companies']}")
        logger.info(f"Total cases: {total_stats['cases']}")
        logger.info(f"Total penalties: {total_stats['penalties']}")
        logger.info(f"Total relationships: {total_stats['relationships']}")
        logger.info(f"Errors: {total_stats['errors']}")
        
        return total_stats

if __name__ == "__main__":
    
    # Initialize Neo4j manager
    with Neo4jManager() as manager:
        # Create constraints
        logger.info("Creating constraints and indexes...")
        manager.create_constraints()
        
        # Initialize graph builder
        builder = GraphBuilder(manager)
        
        # Load from directory (adjust path as needed)
        if len(sys.argv) > 1:
            directory_path = sys.argv[1]
        else:
            directory_path = "./data/kg"  # Default path
        
        logger.info(f"Loading entities from: {directory_path}")
        
        stats = builder.load_from_directory(directory_path, pattern="*.jsonl")
        
        # Get final database stats
        logger.info("\n" + "="*70)
        logger.info("DATABASE STATISTICS")
        logger.info("="*70)
        db_stats = manager.get_database_stats()
        for key, value in db_stats.items():
            logger.info(f"{key}: {value}")
