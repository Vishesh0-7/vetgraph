"""
Simple example demonstrating the refactored VetGraph with provider-agnostic design.

This shows the three main ways to use VetGraph:
1. Using convenience factory methods (recommended)
2. Using provider functions to create custom clients
3. Direct instantiation with an instructor-patched client
"""

import os
from dotenv import load_dotenv

load_dotenv()


def main():
    """Demonstrate basic VetGraph usage with different initialization methods."""
    
    print("🚀 VetGraph Provider-Agnostic Example\n")
    
    # ==================================================================
    # Method 1: Using Convenience Factory Methods (Recommended)
    # ==================================================================
    
    print("=" * 60)
    print("Method 1: Convenience Factory Methods")
    print("=" * 60)
    
    from vetgraph import VetGraph
    
    # Initialize with OpenAI
    vg = VetGraph.from_openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    # Add text and extract knowledge
    text = """
    Albert Einstein was a theoretical physicist who developed the theory of relativity.
    He won the Nobel Prize in Physics in 1921 and worked at Princeton University.
    """
    
    print(f"\n📝 Input text: {text.strip()}\n")
    
    result = vg.add_text(text)
    
    print(f"✅ Extracted {result.get_triple_count()} triples:\n")
    for i, triple in enumerate(result.triples, 1):
        print(f"   {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    # Get graph statistics
    stats = vg.get_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   Nodes: {stats['nodes']}")
    print(f"   Edges: {stats['edges']}")
    print(f"   Density: {stats['density']:.3f}")
    
    # Visualize
    vg.visualize("einstein_graph.html")
    print(f"\n✅ Visualization saved to: einstein_graph.html")
    
    # ==================================================================
    # Method 2: Using Provider Functions (For Custom Configuration)
    # ==================================================================
    
    print("\n" + "=" * 60)
    print("Method 2: Custom Client Configuration")
    print("=" * 60)
    
    from vetgraph import VetGraph, create_openai_client
    
    # Create a custom client with specific settings
    client = create_openai_client(
        api_key=os.getenv("OPENAI_API_KEY"),
        # Can add custom base_url, timeout, etc.
    )
    
    # Create VetGraph with custom client
    vg2 = VetGraph(client=client, model="gpt-4o-mini")
    
    text2 = "Marie Curie discovered radium and polonium."
    result2 = vg2.add_text(text2)
    
    print(f"\n✅ Extracted {result2.get_triple_count()} triples:\n")
    for triple in result2.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    # ==================================================================
    # Method 3: Switching Providers Easily
    # ==================================================================
    
    print("\n" + "=" * 60)
    print("Method 3: Switching Providers (Same API!)")
    print("=" * 60)
    
    # All these providers use the SAME VetGraph API:
    
    # OpenAI
    vg_openai = VetGraph.from_openai(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Anthropic (uncomment if you have API key)
    # vg_anthropic = VetGraph.from_anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Ollama (free, local - uncomment if installed)
    # vg_ollama = VetGraph.from_ollama(model="llama3.2")
    
    # Gemini (uncomment if you have API key)
    # vg_gemini = VetGraph.from_gemini(api_key=os.getenv("GOOGLE_API_KEY"))
    
    text3 = "Python was created by Guido van Rossum."
    result3 = vg_openai.add_text(text3)
    
    print(f"\n✅ Using OpenAI: Extracted {result3.get_triple_count()} triples")
    for triple in result3.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    # ==================================================================
    # Bonus: Schema-Constrained Extraction
    # ==================================================================
    
    print("\n" + "=" * 60)
    print("Bonus: Schema-Constrained Extraction")
    print("=" * 60)
    
    # Define allowed predicates
    schema = ["created", "developed", "invented", "founded"]
    
    text4 = "Steve Jobs founded Apple and created the iPhone."
    print(f"\n📝 Input: {text4}")
    print(f"🔒 Schema: {schema}\n")
    
    result4 = vg_openai.add_text(text4, schema=schema)
    
    print(f"✅ Extracted {result4.get_triple_count()} triples (using only schema predicates):")
    for triple in result4.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    # ==================================================================
    # Summary
    # ==================================================================
    
    print("\n" + "=" * 60)
    print("✨ Summary")
    print("=" * 60)
    
    print("\nVetGraph is now provider-agnostic!")
    print("✅ Same API for OpenAI, Anthropic, Gemini, Ollama, Azure")
    print("✅ Easy to switch providers without changing code")
    print("✅ Powered by the instructor library")
    print("\nThree ways to initialize:")
    print("  1. VetGraph.from_openai(api_key='...')")
    print("  2. VetGraph.from_anthropic(api_key='...')")
    print("  3. VetGraph.from_ollama(model='llama3.2')  # FREE!")
    print("\nSee examples/multi_provider_usage.py for more examples!")
    

if __name__ == "__main__":
    main()
