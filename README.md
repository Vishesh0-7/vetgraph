# VetGraph

**Transform unstructured text into NetworkX knowledge graphs using LLMs**

VetGraph is a provider-agnostic Python library that leverages Large Language Models to extract entities and relationships from unstructured text and construct knowledge graphs using NetworkX.



## Features

- **Provider-Agnostic** — Works with OpenAI, Anthropic (Claude), Google Gemini, Ollama, and Azure OpenAI
- **NetworkX Integration** — Builds standard NetworkX directed graphs for powerful graph analysis
- **Dual Visualization** — Static plots with Matplotlib and interactive HTML visualizations with Pyvis
- **Type Safety** — Pydantic models with Instructor for robust structured outputs
- **Flexible Extraction** — Schema-constrained relationship types and multiple export formats



## Installation

```bash
# Install with OpenAI support
pip install vetgraph[openai]

# Install with Anthropic support
pip install vetgraph[anthropic]

# Install with Gemini support
pip install vetgraph[gemini]

# Install with all providers
pip install vetgraph[all-providers]

# Basic installation (bring your own LLM client)
pip install vetgraph
```



## Quick Start

### OpenAI

```python
from vetgraph import VetGraph

vg = VetGraph.from_openai(api_key="sk-...")

text = """
Albert Einstein was a theoretical physicist who developed the theory of relativity.
He was born in Germany and later moved to the United States. Einstein worked at
Princeton University and won the Nobel Prize in Physics in 1921.
"""

result = vg.add_text(text)
print(f"Extracted {result.get_triple_count()} triples")

vg.visualize("einstein_graph.html")
```

### Anthropic Claude

```python
from vetgraph import VetGraph

vg = VetGraph.from_anthropic(
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022"
)

result = vg.add_text("Marie Curie discovered radium and polonium.")
```

### Google Gemini

```python
from vetgraph import VetGraph

vg = VetGraph.from_gemini(
    api_key="...",
    model="gemini-2.5-flash"
)

result = vg.add_text("The Eiffel Tower is located in Paris, France.")
```

### Ollama (Local, Free)

```python
from vetgraph import VetGraph

vg = VetGraph.from_ollama(model="llama3.2")
result = vg.add_text("Python was created by Guido van Rossum.")
```



## Usage

### Schema-Constrained Extraction

Restrict extracted relationships to a predefined set of types:

```python
from vetgraph import VetGraph

vg = VetGraph.from_openai(api_key="sk-...")

schema = ["works_for", "located_in", "developed_by", "invented"]

result = vg.add_text(
    "Steve Jobs co-founded Apple in Cupertino.",
    schema=schema
)
```

### Graph Analysis with NetworkX

```python
import networkx as nx

graph = vg.get_graph()

centrality = nx.degree_centrality(graph)
print("Most central nodes:",
      sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5])

if nx.has_path(graph, "Einstein", "Princeton"):
    path = nx.shortest_path(graph, "Einstein", "Princeton")
    print("Path:", " -> ".join(path))
```

### Graph Statistics

```python
stats = vg.get_statistics()
print(f"Nodes: {stats['nodes']}")
print(f"Edges: {stats['edges']}")
print(f"Density: {stats['density']:.3f}")
print(f"Connected: {stats['is_connected']}")
print(f"Unique relations: {stats['unique_relations']}")
```

### Visualization

```python
# Interactive HTML visualization (default)
vg.visualize("graph.html")

# Pyvis with custom options
net = vg.visualize_interactive(
    output_file="custom_graph.html",
    height="800px",
    width="100%",
    notebook=False
)

# Matplotlib static plot
vg.visualize_matplotlib(
    figsize=(15, 10),
    node_color="lightcoral",
    node_size=4000,
    save_path="graph.png"
)
```

### Saving and Loading Graphs

```python
# Export in various formats
vg.save_graph("my_graph.json", format="json")           # Node-link JSON
vg.save_graph("my_graph.graphml", format="graphml")     # GraphML (XML)
vg.save_graph("my_graph.gexf", format="gexf")           # GEXF (for Gephi)
vg.save_graph("my_graph.edgelist", format="edgelist")   # Edge list

# Load a previously saved graph
vg.load_graph("my_graph.json", format="json")
```

### Batch Processing

```python
texts = [
    "Leonardo da Vinci painted the Mona Lisa.",
    "The Mona Lisa is displayed in the Louvre Museum.",
    "The Louvre Museum is located in Paris."
]

for text in texts:
    vg.add_text(text)

print(f"Total nodes: {vg.graph.number_of_nodes()}")
print(f"Total edges: {vg.graph.number_of_edges()}")
```

### Custom Provider Configuration

```python
from vetgraph import VetGraph, create_openai_client

client = create_openai_client(
    api_key="sk-...",
    base_url="https://custom-endpoint.com/v1"
)

vg = VetGraph(client=client, model="gpt-4o-mini")
```

### Azure OpenAI

```python
vg = VetGraph.from_azure_openai(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    model="gpt-4-deployment-name",
    api_version="2024-02-01"
)
```



## API Reference

### Factory Methods

| Method | Description |
|--------|-------------|
| `VetGraph.from_openai(api_key, model="gpt-4o-mini")` | Initialize with OpenAI |
| `VetGraph.from_anthropic(api_key, model="claude-3-5-sonnet-20241022")` | Initialize with Anthropic |
| `VetGraph.from_gemini(api_key, model="gemini-2.5-flash")` | Initialize with Google Gemini |
| `VetGraph.from_ollama(model="llama3.2")` | Initialize with Ollama |
| `VetGraph.from_azure_openai(...)` | Initialize with Azure OpenAI |

### Core Methods

| Method | Description |
|--------|-------------|
| `add_text(text, schema=None, temperature=0.3)` | Extract triples from text and add to graph |
| `get_graph()` | Return the underlying NetworkX DiGraph |
| `clear_graph()` | Remove all nodes and edges |
| `get_statistics()` | Return graph statistics as a dictionary |

### Visualization Methods

| Method | Description |
|--------|-------------|
| `visualize(output_path="graph.html")` | Create an interactive HTML visualization |
| `visualize_interactive(...)` | Create a Pyvis visualization with full options |
| `visualize_matplotlib(...)` | Create a static Matplotlib visualization |

### Persistence Methods

| Method | Description |
|--------|-------------|
| `save_graph(output_path, format="json")` | Export graph to file |
| `load_graph(input_path, format="json")` | Load graph from file |



## Supported Models

**OpenAI**
- `gpt-4o-mini` (default, recommended)
- `gpt-4o`
- `gpt-4-turbo`

**Anthropic**
- `claude-3-5-sonnet-20241022` (default, recommended)
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`

**Google Gemini**
- `gemini-2.5-flash` (default, recommended)
- `gemini-2.5-pro`
- `gemini-2.0-flash`

**Ollama**
- `llama3.2` (default)
- `llama3.1`
- `mistral`
- `mixtral`
- Any other locally installed Ollama model


## Dependencies

- [instructor](https://github.com/jxnl/instructor) — Structured outputs for LLMs
- [NetworkX](https://networkx.org/) — Graph construction and analysis
- [Pyvis](https://pyvis.readthedocs.io/) — Interactive graph visualizations
- [Pydantic](https://docs.pydantic.dev/) — Data validation and type safety



## Contributing

Contributions are welcome. Please submit a pull request or open an issue on GitHub.

## License

MIT License. See the [LICENSE](LICENSE) file for details.



## Contact

- **Author:** Vishesh Raju
- **Email:** visheshinus@gmail.com
- **GitHub:** [github.com/Vishesh0-7/vetgraph](https://github.com/Vishesh0-7/vetgraph)