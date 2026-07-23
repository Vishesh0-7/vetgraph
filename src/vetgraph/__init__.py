"""
VetGraph - Transform unstructured text into NetworkX knowledge graphs using LLMs.

VetGraph is a provider-agnostic Python library that leverages Large Language Models
(OpenAI, Anthropic, Gemini, Ollama, etc.) to extract entities and relationships from
unstructured text and construct knowledge graphs using NetworkX. It provides tools
for graph construction, visualization, and analysis.

The library uses the instructor framework to provide a unified interface across
different LLM providers, enabling structured outputs with Pydantic models.

Usage:
    # Using OpenAI (convenience method)
    from vetgraph import VetGraph
    vg = VetGraph.from_openai(api_key="sk-...")
    result = vg.add_text("Einstein developed the theory of relativity.")
    vg.visualize("graph.html")
    
    # Using Anthropic Claude
    vg = VetGraph.from_anthropic(api_key="sk-ant-...")
    
    # Using Ollama (local, free)
    vg = VetGraph.from_ollama(model="llama3.2")
    
    # Advanced: Using custom instructor-patched client
    from vetgraph.providers import create_openai_client
    client = create_openai_client(api_key="sk-...")
    vg = VetGraph(client=client, model="gpt-4o-mini")

"""

__version__ = "1.0.1"
__author__ = "Vishesh Raju"
__email__ = "visheshinus@gmail.com"

from vetgraph.core import VetGraph
from vetgraph.schema import Triple, KnowledgeGraphExtraction
from vetgraph.providers import (
    create_openai_client,
    create_anthropic_client,
    create_gemini_client,
    create_ollama_client,
    create_azure_openai_client,
)

__all__ = [
    "VetGraph",
    "Triple",
    "KnowledgeGraphExtraction",
    "create_openai_client",
    "create_anthropic_client",
    "create_gemini_client",
    "create_ollama_client",
    "create_azure_openai_client",
]
