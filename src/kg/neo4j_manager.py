"""Neo4j database manager for knowledge graph operations"""

import os
from loguru import logger
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Neo4jConfig(BaseSettings):
    """Configuration for Neo4j connection.
    
    Reads from environment variables with NEO4J_ prefix.
    
    Example .env:
        NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
        NEO4J_USERNAME=neo4j
        NEO4J_PASSWORD=your-password
    """

    uri: str = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    username: str = os.getenv("NEO4J_USERNAME") or "neo4j"
    password: str = os.getenv("NEO4J_PASSWORD") or "password"
    database: str = os.getenv("NEO4J_DATABASE") or "neo4j"

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

class Neo4jManager:
    """Manages Neo4j database connection and operations"""

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """Initialize Neo4j manager.
        
        Args:
            config: Neo4jConfig object (loads from .env if not provided)
        """
        self.config = config or Neo4jConfig()
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            logger.info(f"Connecting to Neo4j at {self.config.uri}")
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password)
            )
            self.driver.verify_connectivity()
            logger.info(" Successfully connected to Neo4j")
        except AuthError as e:
            logger.error(f" Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f" Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f" Connection failed: {e}")
            raise
    
    def _close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._close()
    
    def verify_connection(self) -> bool:
        """Verify Neo4j connection is working.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run("RETURN 1 AS TEST")
                return result.single()["TEST"] == 1
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False
    
    def execute_query(
            self,
            query: str,
            parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, parameters or {})
            return [
                record.data()
                for record in result
            ]
    
    def execute_write(
            self,
            query: str,
            parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a write query (CREATE, MERGE, etc.)
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query summary statistics
        """
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, parameters or {})
            summary = result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created,
                "properties_set": summary.counters.properties_set,
                "labels_added": summary.counters.labels_added,
            }
    
    def clear_database(self):
        """Clear all nodes and relationships from database"""
        logger.warning(" Clearing entire Neo4j database...")
        query = "MATCH (n) DETACH DELETE n"
        result = self.execute_write(query)
        logger.info(f" Database cleared. Deleted nodes: {result.get('nodes_deleted', 0)}")
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics
        
        Returns:
            Dictionary with node and relationship counts
        """
        stats = {}
        
        # Count nodes by label
        node_query = """
        MATCH (n)
        RETURN labels(n) AS labels, count(n) AS count
        """
        node_results = self.execute_query(node_query)
        
        for record in node_results:
            label = record["labels"][0] if record["labels"] else "Unknown"
            stats[f"{label}_nodes"] = record["count"]
        
        # Count total nodes
        total_nodes_query = "MATCH (n) RETURN count(n) AS count"
        total_nodes = self.execute_query(total_nodes_query)[0]["count"]
        stats["total_nodes"] = total_nodes
        
        # Count relationships
        rel_query = "MATCH ()-[r]->() RETURN count(r) AS count"
        total_rels = self.execute_query(rel_query)[0]["count"]
        stats["total_relationships"] = total_rels
        
        return stats
    
    def create_constraints(self):
        """Create uniqueness constraints and indexes for better performance"""
        constraints = [
            # Person node: unique by name
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            
            # Company node: unique by name
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            
            # Case node: unique by LR number
            "CREATE CONSTRAINT case_lr_number IF NOT EXISTS FOR (c:Case) REQUIRE c.lr_number IS UNIQUE",
            
            # Penalty node: composite index
            "CREATE INDEX penalty_type_amount IF NOT EXISTS FOR (p:Penalty) ON (p.penalty_type, p.amount)",
        ]
        
        logger.info("Creating constraints and indexes...")
        for constraint in constraints:
            try:
                self.execute_write(constraint)
                logger.info(f" {constraint.split()[1]}")
            except Exception as e:
                # Constraint might already exist
                logger.debug(f"Constraint/index might already exist: {e}")
        
        logger.info(" All constraints and indexes created")
    
    def get_node_count(self, label: str) -> int:
        """Get count of nodes with specific label
        
        Args:
            label: Node label (e.g., 'Person', 'Company')
            
        Returns:
            Number of nodes
        """
        query = f"MATCH (n:{label}) RETURN count(n) AS count"
        result = self.execute_query(query)
        return result[0]["count"] if result else 0
    
    def get_relationship_count(self, rel_type: Optional[str] = None) -> int:
        """Get count of relationships
        
        Args:
            rel_type: Relationship type (e.g., 'CHARGED_IN'). If None, counts all
            
        Returns:
            Number of relationships
        """
        if rel_type:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"
        
        result = self.execute_query(query)
        return result[0]["count"] if result else 0

if __name__ == "__main__":
    
    logger.info("=" * 70)
    logger.info("NEO4J CONNECTION TEST")
    logger.info("=" * 70)
    
    try:
        # Initialize manager (reads from .env)
        with Neo4jManager() as manager:
            logger.info("\n Testing connection...")
            if manager.verify_connection():
                logger.info("   Connection successful!")
            else:
                logger.error("   Connection failed!")
                exit(1)
            
            logger.info("\n Creating constraints and indexes...")
            manager.create_constraints()
            
            logger.info("\n Getting database statistics...")
            stats = manager.get_database_stats()
            logger.info("   Database Stats:")
            for key, value in stats.items():
                logger.info(f"   - {key}: {value}")
            
            logger.info("\n Testing simple query...")
            result = manager.execute_query("RETURN 'Hello Neo4j!' AS message")
            logger.info(f"   Query result: {result[0]['message']}")
            
            logger.info("\n" + "=" * 70)
            logger.info(" ALL TESTS PASSED!")
            logger.info("=" * 70)
    
    except Exception as e:
        logger.error(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
