"""Graph builder to load extracted entities into Neo4j knowledge graph"""

import os
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
            p.created_at = dateime()
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
