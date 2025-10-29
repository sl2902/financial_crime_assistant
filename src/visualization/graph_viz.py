"""Network graph visualization using pyvis with emoji icons."""

import os
import tempfile
from typing import Any, List, Dict, Optional
from pathlib import Path
from loguru import logger

try:
    from pyvis.network import Network
except ImportError:
    raise ImportError("Please install pyvis: pip install pyvis")

class GraphVisualizer:
    """Create interactive network graphs from Neo4j query results"""

    NODE_CONFIG = {
        "Person": {
            "icon": "👤",
            "color": "#ff6b6b",
            "size": 30,
            "title_prefix": "Person"
        },
        "Company": {
            "icon": "🏢",
            "color": "#4ecdc4",
            "size": 30,
            "title_prefix": "Company"
        },
        "Case": {
            "icon": "⚖️",
            "color": "#ffe66d",
            "size": 35,
            "title_prefix": "Case"
        },
        "Penalty": {
            "icon": "💰",
            "color": "#95e1d3",
            "size": 25,
            "title_prefix": "Penalty"
        }
    }

    EDGE_CONFIG = {
        "WORKED_AT": {
            "icon": "💼",
            "label": "worked at",
            "color": "#888888",
            "title": "Employment relationship"
        },
        "CHARGED_IN": {
            "icon": "⚠️",
            "label": "charged in",
            "color": "#ff6b6b",
            "title": "Charged in case"
        },
        "INVOLVED_IN": {
            "icon": "🔗",
            "label": "involved in",
            "color": "#4ecdc4",
            "title": "Company involved in case"
        },
        "HAS_PENALTY": {
            "icon": "💸",
            "label": "penalty",
            "color": "#ffe66d",
            "title": "Case has penalty"
        },
        "RECEIVED_PENALTY": {
            "icon": "💵",
            "label": "received",
            "color": "#95e1d3",
            "title": "Received penalty"
        }
    }

    def __init__(
            self,
            height: str = "650px",
            width: str = "100%",
            bgcolor: str = "#1a1a1a",
            font_color: str = "white",
            layout: str = "barnes_hut"
        ):
        """Initialize graph visualizer.
        
        Args:
            height: Graph height
            width: Graph width
            bgcolor: Background color
            font_color: Font color
            layout: Layout algorithm ('barnes_hut', 'force_atlas', 'repulsion', 'hierarchical')
        """
        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color
        self.layout = layout
    
    def create_network(self) -> Network:
        """Create a pyvis Network with configured settings.
        
        Returns:
            Configured Network instance
        """
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            select_menu=True,
            filter_menu=True,
            notebook=False
        )

        if self.layout == "barnes_hut":
            # Configure Physics for layout
            net.barnes_hut(
                gravity=-8000,
                central_gravity=0.3,
                spring_length=150,
                spring_strength=0.001,
                damping=0.09,
                overlap=0
            )
        
        elif self.layout == "repulsion":
            net.repulsion(
                node_distance=150,
                central_gravity=0.2,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09
            )
    
        elif self.layout == "hierarchical":
            net.set_options("""
            {
            "layout": {
                "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 150,
                "nodeSpacing": 200,
                "treeSpacing": 200
                }
            },
            "physics": {
                "enabled": false
            }
            }
            """)
        
        # Enable drag and zoom
        net.set_options("""
        {
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 200
            }
          }
        }
        """)

        return net
    
    def parse_neo4j_results(self, results: List[Dict[str, Any]] | Dict[str, Any]) -> Dict[str, List]:
        """Parse Neo4j query results into nodes and edges
        
        Args:
            results: List of Neo4j query result dictionaries
            
        Returns:
            Dictionary with 'nodes' and 'edges' lists
        """
        nodes = {}
        edges = []

        if isinstance(results, dict):
            if 'results' in results:
                records = results['results']  # Current format
            elif 'success' in results and 'results' in results:
                records = results['results']
            else:
                records = [results]  # Single record
        elif isinstance(results, list):
            records = results
        else:
            logger.error(f"Unexpected results type: {type(results)}")
            return {'nodes': [], 'edges': []}
        
        logger.info(f"Parsing {len(records)} records for visualization")
            
        for i, record in enumerate(records):
            logger.debug(f"Record {i+1}: keys={list(record.keys())}")
            # === EXTRACT NODES ===

            lr_number = record.get('case_lr_number') or record.get('lr_number')
            if lr_number and lr_number not in nodes:
                title = record.get('case_title', '') or record.get('title', '')
                nodes[lr_number] = {
                    'id': lr_number,
                    'name': lr_number,
                    'type': 'Case',
                    'details': title
                }
                logger.debug(f"  Added case: {lr_number}")
            
            persons = record.get('persons', []) or record.get('defendants', [])
            if isinstance(persons, list):
                for person in persons:
                    if person and person not in nodes:
                        nodes[person] = {
                            'id': person,
                            'name': person,
                            'type': 'Person',
                            'details': 'Defendant'
                        }
                        logger.debug(f"  Added person: {person}")
            
            # Company nodes from 'companies' list
            companies = record.get('companies', [])
            if isinstance(companies, list):
                for company in companies:
                    if company and company not in nodes:
                        nodes[company] = {
                            'id': company,
                            'name': company,
                            'type': 'Company',
                            'details': 'Involved entity'
                        }
                        logger.debug(f"  Added company: {company}")
            
            # === CREATE EDGES ===
            
            # Person -> Case (CHARGED_IN)
            if lr_number and persons:
                for person in persons:
                    if person:
                        edges.append({
                            'source': person,
                            'target': lr_number,
                            'relationship': 'CHARGED_IN'
                        })
                        logger.debug(f"  Edge: {person} -> {lr_number}")
            
            # Company -> Case (INVOLVED_IN)
            if lr_number and companies:
                for company in companies:
                    if company:
                        edges.append({
                            'source': company,
                            'target': lr_number,
                            'relationship': 'INVOLVED_IN'
                        })
                        logger.debug(f"  Edge: {company} -> {lr_number}")
        
        # Remove duplicate edges
        seen = set()
        unique_edges = []
        for edge in edges:
            edge_key = (edge['source'], edge['target'], edge['relationship'])
            if edge_key not in seen:
                seen.add(edge_key)
                unique_edges.append(edge)
        
        logger.info(f" Parsed {len(nodes)} nodes and {len(unique_edges)} edges")
        
        if not nodes:
            logger.warning(" NO NODES CREATED - check data format!")
        if not unique_edges:
            logger.warning(" NO EDGES CREATED - check relationship logic!")
        
        return {
            'nodes': list(nodes.values()),
            'edges': unique_edges
        }
    
    def add_nodes(self, net: Network, nodes: List[Dict[str, Any]]):
        """Add nodes to the network
        
        Args:
            net: pyvis Network instance
            nodes: List of node dictionaries
        """
        for node in nodes:
            node_type = node['type']
            config = self.NODE_CONFIG.get(node_type, self.NODE_CONFIG['Person'])
            
            # Create label with emoji icon
            label = f"{config['icon']} {node['name']}"
            
            # Create hover tooltip
            title = f"{config['title_prefix']}: {node['name']}"
            if node.get('details'):
                title += f"\n{node['details']}"
            
            net.add_node(
                node['id'],
                label=label,
                title=title,
                color=config['color'],
                size=config['size'],
                font={'size': 14, 'color': 'white', 'face': 'Arial'}
            )
    
    def add_edges(self, net: Network, edges: List[Dict[str, Any]]):
        """Add edges to the network
        
        Args:
            net: pyvis Network instance
            edges: List of edge dictionaries
        """
        for edge in edges:
            rel_type = edge['relationship']
            config = self.EDGE_CONFIG.get(rel_type, {
                'icon': '🔗',
                'label': rel_type,
                'color': '#888888',
                'title': rel_type
            })
            
            # Create edge label with emoji
            edge_label = f"{config['icon']} {config['label']}"
            
            net.add_edge(
                edge['source'],
                edge['target'],
                label=edge_label,
                title=config['title'],
                color=config['color'],
                width=2,
                arrows='to',
                smooth={'type': 'curvedCW', 'roundness': 0.2}
            )
    
    def visualize(
        self,
        results: List[Dict[str, Any]] | Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Create network visualization from Neo4j results.
        
        Args:
            results: Neo4j query results
            output_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        # Parse results
        graph_data = self.parse_neo4j_results(results)
        
        # Check if we have data to visualize
        if not graph_data['nodes']:
            # Return empty graph message
            net = self.create_network()
            net.add_node(
                "empty",
                label="📊 No connections found",
                color="#888888",
                size=40,
                font={'size': 16, 'color': 'white'}
            )
            if output_path is None:
                output_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.html'
                ).name
            net.save_graph(output_path)
            return output_path
        
        # Check if we have edges (real relationships from Neo4j)
        if not graph_data['edges'] and len(graph_data['nodes']) > 1:
            # We have nodes but no explicit edges
            # This means the query didn't return relationship data
            # Show nodes only with a warning
            net = self.create_network()
            for node in graph_data['nodes']:
                config = self.NODE_CONFIG.get(node['type'], self.NODE_CONFIG['Person'])
                net.add_node(
                    node['id'],
                    label=f"{config['icon']} {node['name']}",
                    title=f"{config['title_prefix']}: {node['name']}",
                    color=config['color'],
                    size=config['size']
                )
            if output_path is None:
                output_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.html'
                ).name
            net.save_graph(output_path)
            return output_path
        
        # Create network with nodes and edges
        net = self.create_network()
        
        # Add nodes and edges
        self.add_nodes(net, graph_data['nodes'])
        self.add_edges(net, graph_data['edges'])
        
        # Save to file
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix='.html'
            ).name
        
        net.save_graph(output_path)
        
        return output_path
    
    def visualize_from_graph_tool_result(
        self,
        graph_tool_result: str,
        output_path: Optional[str] = None
    ) -> str:
        """Create visualization from graph tool text result.
        
        Args:
            graph_tool_result: Text output from graph_agent_tool
            output_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        # Simple parser for graph tool results
        # Example format: "1. person_name: John; companies: Acme, Inc; cases: LR-123"
        
        nodes = []
        edges = []
        lines = graph_tool_result.split('\n')
        
        for line in lines:
            if not line.strip() or line.startswith('Found') or line.startswith('KNOWLEDGE'):
                continue
            
            # Extract data from line (basic parsing)
            # This is a simple implementation - adjust based on actual format
            if 'person_name:' in line.lower():
                parts = line.split(';')
                person_name = None
                companies_list = []
                cases_list = []
                
                for part in parts:
                    if 'person_name:' in part.lower():
                        person_name = part.split(':')[1].strip()
                        nodes.append({
                            'id': person_name,
                            'name': person_name,
                            'type': 'Person',
                            'details': ''
                        })
                    elif 'companies:' in part.lower():
                        companies_str = part.split(':')[1].strip()
                        companies_list = [c.strip() for c in companies_str.split(',')]
                        for company in companies_list:
                            if company:
                                nodes.append({
                                    'id': company,
                                    'name': company,
                                    'type': 'Company',
                                    'details': ''
                                })
                    elif 'cases:' in part.lower():
                        cases_str = part.split(':')[1].strip()
                        cases_list = [c.strip() for c in cases_str.split(',')]
                        for case in cases_list:
                            if case:
                                nodes.append({
                                    'id': case,
                                    'name': case,
                                    'type': 'Case',
                                    'details': ''
                                })
                
                # Create edges
                if person_name:
                    for company in companies_list:
                        if company:
                            edges.append({
                                'source': person_name,
                                'target': company,
                                'relationship': 'WORKED_AT'
                            })
                    for case in cases_list:
                        if case:
                            edges.append({
                                'source': person_name,
                                'target': case,
                                'relationship': 'CHARGED_IN'
                            })
        
        # Create visualization with extracted data
        graph_data = {'nodes': nodes, 'edges': edges}
        
        return self.visualize(graph_data, output_path)


# Convenience function for quick usage
def create_graph_visualization(
    neo4j_results: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> str:
    """Quick function to create graph visualization.
    
    Args:
        neo4j_results: Results from Neo4j query
        output_path: Optional output file path
        
    Returns:
        Path to generated HTML file
    
    Example:
        >>> from src.kg import Neo4jManager, GraphQueryTool
        >>> manager = Neo4jManager()
        >>> tool = GraphQueryTool(manager)
        >>> results = tool.query("person_info", name="John Smith")
        >>> html_path = create_graph_visualization(results['persons'])
        >>> print(f"Graph saved to: {html_path}")
    """
    viz = GraphVisualizer()
    return viz.visualize(neo4j_results, output_path)


if __name__ == "__main__":
    # Example usage
    print("Graph Visualization Module")
    print("=" * 60)
    
    # Example data
    example_results = [
        {
            'person_name': 'John Smith',
            'companies': ['Acme Corp', 'XYZ Inc'],
            'lr_number': 'LR-26163',
            'title': 'Securities Fraud',
            'total_penalty': 5000000
        },
        {
            'person_name': 'Jane Doe',
            'companies': ['Acme Corp'],
            'lr_number': 'LR-26163',
            'title': 'Securities Fraud',
            'total_penalty': 5000000
        }
    ]
    
    # Create visualization
    viz = GraphVisualizer(layout="force_atlas")
    output_file = viz.visualize(example_results, "example_graph.html")
    
    print(f"✅ Graph visualization created: {output_file}")
    print("\nOpen the file in a browser to see the interactive graph!")
    print("\nFeatures:")
    print("  - 👤 Red nodes = Persons")
    print("  - 🏢 Teal nodes = Companies")
    print("  - ⚖️ Yellow nodes = Cases")
    print("  - 💰 Mint nodes = Penalties")
    print("  - Hover over nodes/edges for details")
    print("  - Drag nodes to rearrange")
    print("  - Zoom with mouse wheel")
