"""
Core functionality for building knowledge graphs from unstructured text using LLMs.

This module provides the main VetGraph class for extracting entities and relationships
from text and constructing NetworkX graphs with visualization capabilities.

The VetGraph class is provider-agnostic and works with any LLM that has been patched
with the instructor library (OpenAI, Anthropic, Gemini, Ollama, etc.).
"""

import json
from typing import Any, Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from networkx.readwrite import json_graph
import instructor

from vetgraph.schema import Triple, KnowledgeGraphExtraction


class VetGraph:
    """Provider-agnostic knowledge graph builder using LLMs.
    
    VetGraph works with any LLM provider (OpenAI, Anthropic, Gemini, Ollama, etc.) 
    by accepting an instructor-patched client. This allows you to extract entities 
    and relationships as triples (subject-predicate-object) and construct NetworkX 
    directed graphs that can be analyzed and visualized.
    
    Attributes:
        graph: NetworkX directed graph storing the knowledge graph
        client: Instructor-patched LLM client for structured outputs
        model: Name of the LLM model to use for extraction
    
    Examples:
        Using OpenAI:
            >>> from vetgraph import VetGraph
            >>> vg = VetGraph.from_openai(api_key="sk-...")
            >>> result = vg.add_text("Einstein developed relativity")
        
        Using Anthropic:
            >>> vg = VetGraph.from_anthropic(api_key="sk-ant-...")
        
        Using Gemini:
            >>> vg = VetGraph.from_gemini(api_key="...")
        
        Using Ollama (local, free):
            >>> vg = VetGraph.from_ollama(model="llama3.2")
    """
    
    def __init__(
        self,
        client: instructor.Instructor,
        model: Optional[str] = "gpt-4o-mini",
    ):
        """
        Initialize VetGraph with an instructor-patched LLM client.
        
        Args:
            client: Instructor-patched LLM client (from create_*_client functions)
            model: Model name to use for extraction (provider-specific).
                   Set to None for providers like Gemini where model is configured in the client.
        
        Note:
            For convenience, use the factory methods VetGraph.from_openai(),
            VetGraph.from_anthropic(), VetGraph.from_ollama(), etc.
        """
        self.graph = nx.DiGraph()
        self.client = client
        self.model = model
    
    @classmethod
    def from_openai(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ) -> "VetGraph":
        """
        Create a VetGraph instance using OpenAI.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini)
            base_url: Optional base URL for OpenAI API
        
        Returns:
            VetGraph instance configured for OpenAI
        """
        from vetgraph.providers import create_openai_client
        client = create_openai_client(api_key=api_key, base_url=base_url)
        return cls(client=client, model=model)
    
    @classmethod
    def from_anthropic(
        cls,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> "VetGraph":
        """
        Create a VetGraph instance using Anthropic Claude.
        
        Args:
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
        
        Returns:
            VetGraph instance configured for Anthropic
        """
        from vetgraph.providers import create_anthropic_client
        client = create_anthropic_client(api_key=api_key)
        return cls(client=client, model=model)
    
    @classmethod
    def from_gemini(
        cls,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
    ) -> "VetGraph":
        """
        Create a VetGraph instance using Google Gemini.
        
        Args:
            api_key: Google API key (if None, uses GOOGLE_API_KEY env var)
            model: Gemini model to use (default: gemini-2.5-flash)
                   Valid models: gemini-3-pro, gemini-3-flash, gemini-2.5-pro,
                   gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash
            temperature: Temperature for generation (0.0-2.0, default: 0.3)
        
        Returns:
            VetGraph instance configured for Gemini
        
        Note:
            For Gemini, the model and temperature are configured in the client.
        """
        from vetgraph.providers import create_gemini_client
        client = create_gemini_client(api_key=api_key, model=model, temperature=temperature)
        return cls(client=client, model=None)
    
    @classmethod
    def from_ollama(
        cls,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
    ) -> "VetGraph":
        """
        Create a VetGraph instance using Ollama (local, free LLM).
        
        Args:
            model: Ollama model to use (default: llama3.2)
            base_url: Ollama API base URL (default: http://localhost:11434/v1)
        
        Returns:
            VetGraph instance configured for Ollama
        
        Note:
            Requires Ollama to be installed and running locally.
            Download from: https://ollama.ai
        """
        from vetgraph.providers import create_ollama_client
        client = create_ollama_client(base_url=base_url)
        return cls(client=client, model=model)
    
    @classmethod
    def from_azure_openai(
        cls,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        model: str = "gpt-4",
        api_version: str = "2024-02-01",
    ) -> "VetGraph":
        """
        Create a VetGraph instance using Azure OpenAI.
        
        Args:
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL
            model: Deployment name in Azure
            api_version: Azure OpenAI API version
        
        Returns:
            VetGraph instance configured for Azure OpenAI
        """
        from vetgraph.providers import create_azure_openai_client
        client = create_azure_openai_client(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return cls(client=client, model=model)
    
    def _get_extraction_prompt(self, text: str, schema: Optional[List[str]] = None) -> str:
        """
        Generate the system prompt for knowledge extraction.
        
        Args:
            text: The input text to analyze
            schema: Optional list of allowed predicates
        
        Returns:
            System prompt string for the LLM
        """
        base_prompt = (
            "You are an expert linguistic analyst specializing in knowledge graph construction. "
            "Your task is to extract entities and their directional relationships from text.\n\n"
            "Extract knowledge as triples in the form: (subject, predicate, object)\n"
            "- Subject: The entity performing an action or being described\n"
            "- Predicate: The relationship or action connecting subject and object\n"
            "- Object: The entity receiving the action or being related to the subject\n\n"
            "Guidelines:\n"
            "- Use clear, concise entity names\n"
            "- Predicates should be lowercase and use underscores (e.g., 'works_for', 'located_in')\n"
            "- Capture directional relationships (from subject to object)\n"
            "- Extract all meaningful relationships, including implicit ones\n"
        )
        
        if schema:
            schema_list = ", ".join(f"'{p}'" for p in schema)
            base_prompt += (
                f"\n**IMPORTANT**: You MUST use only these predicates: {schema_list}\n"
                f"Do not create any predicates outside this list."
            )
        else:
            base_prompt += (
                "\n- Create predicates that accurately describe the relationship type\n"
                "- Be consistent with predicate naming across similar relationships"
            )
        
        return base_prompt
    
    def add_text(
        self,
        text: str,
        schema: Optional[List[str]] = None,
        temperature: float = 0.3,
    ) -> KnowledgeGraphExtraction:
        """
        Extract knowledge triples from text and add them to the graph.
        
        Args:
            text: Input text to extract knowledge from
            schema: Optional list of allowed predicates for schema-constrained extraction
            temperature: Sampling temperature for the LLM (0.0-2.0)
        
        Returns:
            KnowledgeGraphExtraction object containing all extracted triples
        
        Raises:
            ValueError: If the API call fails or extraction returns invalid data
        """
        system_prompt = self._get_extraction_prompt(text, schema)
        
        create_params = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract knowledge triples from the following text:\n\n{text}"}
            ],
            "response_model": KnowledgeGraphExtraction,
        }
        
        # Only add model and temperature if model is set
        if self.model is not None:
            create_params["model"] = self.model
            create_params["temperature"] = temperature
        
        extraction = self.client.chat.completions.create(**create_params)
        
        if extraction is None:
            raise ValueError("Failed to parse extraction from LLM response")
        
        # Add each triple to the graph as an edge
        for triple in extraction.triples:
            self.graph.add_edge(
                triple.subject,
                triple.object,
                relation=triple.predicate
            )
        
        return extraction
    
    def get_graph(self) -> nx.DiGraph:
        """Get the current NetworkX graph."""
        return self.graph
    
    def clear_graph(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.graph.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": 0.0,
            "is_connected": False,
            "unique_relations": len(set(
                data.get("relation", "unknown") 
                for _, _, data in self.graph.edges(data=True)
            ))
        }
        
        if stats["nodes"] > 1:
            stats["density"] = stats["edges"] / (stats["nodes"] * (stats["nodes"] - 1))
            stats["is_connected"] = nx.is_weakly_connected(self.graph)
        
        return stats
    
    def visualize_matplotlib(
        self,
        figsize: tuple = (12, 8),
        node_color: str = "lightblue",
        node_size: int = 3000,
        font_size: int = 10,
        with_labels: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the graph using matplotlib.
        
        Args:
            figsize: Figure size as (width, height)
            node_color: Color for nodes
            node_size: Size of nodes
            font_size: Font size for labels
            with_labels: Whether to show node labels
            save_path: Optional path to save the figure
        """
        if self.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty, nothing to visualize")
            return
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_color, node_size=node_size, alpha=0.9
        )
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, arrows=True, arrowsize=20)
        
        if with_labels:
            labels = {node: node for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=font_size)
        
        edge_labels = nx.get_edge_attributes(self.graph, "relation")
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)
        
        plt.axis("off")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def visualize_interactive(
        self,
        output_file: str = "knowledge_graph.html",
        height: str = "750px",
        width: str = "100%",
        notebook: bool = False,
    ) -> Network:
        """
        Create an interactive visualization using pyvis.
        
        Args:
            output_file: Path to save the HTML file
            height: Height of the visualization
            width: Width of the visualization
            notebook: Whether to render in Jupyter notebook
            
        Returns:
            Pyvis Network object
        """
        if self.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty, nothing to visualize")
            return Network(height=height, width=width, directed=True, notebook=notebook)
        
        net = Network(height=height, width=width, directed=True, notebook=notebook)
        
        for node in self.graph.nodes():
            net.add_node(node, label=node, title=node)
        
        for source, target, attrs in self.graph.edges(data=True):
            edge_relation = attrs.get("relation", "related_to")
            net.add_edge(source, target, title=edge_relation, label=edge_relation)
        
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          },
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "font": {"size": 10, "align": "middle"}
          }
        }
        """)
        
        if notebook:
            return net
        else:
            net.save_graph(output_file)
            return net
    
    def visualize(self, output_path: str = 'graph.html') -> None:
        """
        Generate an interactive HTML visualization of the current graph.
        
        Args:
            output_path: Path where the HTML file will be saved
        """
        if self.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty, nothing to visualize")
            return
        
        net = Network(height='750px', width='100%', directed=True, notebook=False)
        
        for node in self.graph.nodes():
            net.add_node(node, label=node, title=node)
        
        for source, target, attrs in self.graph.edges(data=True):
            relation = attrs.get('relation', 'related_to')
            net.add_edge(source, target, title=relation, label=relation)
        
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95
            }
          },
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "font": {"size": 10, "align": "middle"},
            "smooth": {"type": "continuous"}
          },
          "nodes": {
            "font": {"size": 14}
          }
        }
        """)
        
        net.save_graph(output_path)
        print(f"✅ Graph visualization saved to: {output_path}")
    
    def save_graph(self, output_path: str = 'graph.json', format: str = 'json') -> None:
        """
        Export the NetworkX graph data to a file.
        
        Args:
            output_path: Path where the graph will be saved
            format: Export format ('json', 'graphml', 'gexf', 'edgelist')
        
        Raises:
            ValueError: If an unsupported format is specified
        """
        if self.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty, nothing to save")
            return
        
        format = format.lower()
        
        if format == 'json':
            data = json_graph.node_link_data(self.graph)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Graph saved to: {output_path} (JSON format)")
        elif format == 'graphml':
            nx.write_graphml(self.graph, output_path)
            print(f"✅ Graph saved to: {output_path} (GraphML format)")
        elif format == 'gexf':
            nx.write_gexf(self.graph, output_path)
            print(f"✅ Graph saved to: {output_path} (GEXF format)")
        elif format == 'edgelist':
            nx.write_edgelist(self.graph, output_path, data=True)
            print(f"✅ Graph saved to: {output_path} (Edge list format)")
        else:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: 'json', 'graphml', 'gexf', 'edgelist'"
            )
    
    def load_graph(self, input_path: str, format: str = 'json') -> None:
        """
        Load a previously saved graph from a file.
        
        Args:
            input_path: Path to the graph file
            format: Format of the file ('json', 'graphml', 'gexf', 'edgelist')
        
        Raises:
            ValueError: If an unsupported format is specified
            FileNotFoundError: If the file doesn't exist
        """
        format = format.lower()
        
        if format == 'json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.graph = json_graph.node_link_graph(data, directed=True)
            print(f"✅ Graph loaded from: {input_path} (JSON format)")
        elif format == 'graphml':
            self.graph = nx.read_graphml(input_path)
            if not isinstance(self.graph, nx.DiGraph):
                self.graph = nx.DiGraph(self.graph)
            print(f"✅ Graph loaded from: {input_path} (GraphML format)")
        elif format == 'gexf':
            self.graph = nx.read_gexf(input_path)
            if not isinstance(self.graph, nx.DiGraph):
                self.graph = nx.DiGraph(self.graph)
            print(f"✅ Graph loaded from: {input_path} (GEXF format)")
        elif format == 'edgelist':
            self.graph = nx.read_edgelist(input_path, create_using=nx.DiGraph())
            print(f"✅ Graph loaded from: {input_path} (Edge list format)")
        else:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: 'json', 'graphml', 'gexf', 'edgelist'"
            )
