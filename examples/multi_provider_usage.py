"""
Multi-Provider Usage Examples for VetGraph

This example demonstrates how to use VetGraph with different LLM providers:
- OpenAI GPT models
- Anthropic Claude models
- Ollama (local, free)
- Azure OpenAI
- Google Gemini

All providers use the same VetGraph API thanks to the instructor library.
"""

import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()


def example_openai():
    """Example using OpenAI GPT models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Using OpenAI GPT")
    print("=" * 60)
    
    from vetgraph import VetGraph
    
    # Method 1: Using convenience factory method
    vg = VetGraph.from_openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    text = "Albert Einstein developed the theory of relativity."
    result = vg.add_text(text)
    
    print(f"\n✅ Extracted {result.get_triple_count()} triples:")
    for triple in result.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    print(f"\n📊 Graph has {vg.graph.number_of_nodes()} nodes and {vg.graph.number_of_edges()} edges")


def example_anthropic():
    """Example using Anthropic Claude models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Using Anthropic Claude")
    print("=" * 60)
    
    from vetgraph import VetGraph
    
    # Using convenience factory method
    vg = VetGraph.from_anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-sonnet-20241022"
    )
    
    text = "Marie Curie discovered radium and won two Nobel Prizes."
    result = vg.add_text(text)
    
    print(f"\n✅ Extracted {result.get_triple_count()} triples:")
    for triple in result.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    print(f"\n📊 Graph has {vg.graph.number_of_nodes()} nodes and {vg.graph.number_of_edges()} edges")


def example_ollama():
    """Example using Ollama (local, free LLM)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Using Ollama (Local, Free)")
    print("=" * 60)
    
    from vetgraph import VetGraph
    
    # No API key needed for Ollama!
    vg = VetGraph.from_ollama(model="llama3.2")
    
    text = "Python was created by Guido van Rossum in 1991."
    result = vg.add_text(text)
    
    print(f"\n✅ Extracted {result.get_triple_count()} triples:")
    for triple in result.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    print(f"\n📊 Graph has {vg.graph.number_of_nodes()} nodes and {vg.graph.number_of_edges()} edges")
    print("\n💡 Tip: Install Ollama from https://ollama.ai")


def example_azure_openai():
    """Example using Azure OpenAI."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Using Azure OpenAI")
    print("=" * 60)
    
    from vetgraph import VetGraph
    
    # Azure OpenAI requires endpoint and deployment name
    vg = VetGraph.from_azure_openai(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model="gpt-4"  # Your deployment name in Azure
    )
    
    text = "Shakespeare wrote Romeo and Juliet in 1595."
    result = vg.add_text(text)
    
    print(f"\n✅ Extracted {result.get_triple_count()} triples:")
    for triple in result.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    print(f"\n📊 Graph has {vg.graph.number_of_nodes()} nodes and {vg.graph.number_of_edges()} edges")


def example_gemini():
    """Example using Google Gemini."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Using Google Gemini")
    print("=" * 60)
    
    from vetgraph import VetGraph
    
    vg = VetGraph.from_gemini(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-1.5-flash"
    )
    
    text = "Tesla was founded by Elon Musk and produces electric vehicles."
    result = vg.add_text(text)
    
    print(f"\n✅ Extracted {result.get_triple_count()} triples:")
    for triple in result.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    print(f"\n📊 Graph has {vg.graph.number_of_nodes()} nodes and {vg.graph.number_of_edges()} edges")


def example_custom_client():
    """Example using a custom instructor-patched client."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Using Custom Client (Advanced)")
    print("=" * 60)
    
    from vetgraph import VetGraph, create_openai_client
    
    # Create a custom client with specific settings
    client = create_openai_client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=None,  # Can use custom base URL
    )
    
    # Create VetGraph with the custom client
    vg = VetGraph(client=client, model="gpt-4o-mini")
    
    text = "The Eiffel Tower was completed in 1889 in Paris."
    result = vg.add_text(text)
    
    print(f"\n✅ Extracted {result.get_triple_count()} triples:")
    for triple in result.triples:
        print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    print(f"\n📊 Graph has {vg.graph.number_of_nodes()} nodes and {vg.graph.number_of_edges()} edges")


def example_switching_providers():
    """Example showing how to switch between providers seamlessly."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Switching Between Providers")
    print("=" * 60)
    
    from vetgraph import VetGraph
    
    text = "NASA launched Apollo 11 in 1969."
    
    # Try with different providers - same API!
    providers = [
        ("OpenAI", VetGraph.from_openai(api_key=os.getenv("OPENAI_API_KEY"))),
        # Uncomment to test other providers:
        # ("Anthropic", VetGraph.from_anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))),
        # ("Ollama", VetGraph.from_ollama()),
    ]
    
    for provider_name, vg in providers:
        try:
            result = vg.add_text(text)
            print(f"\n✅ {provider_name}: Extracted {result.get_triple_count()} triples")
            for triple in result.triples:
                print(f"   • ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
        except Exception as e:
            print(f"\n❌ {provider_name} failed: {e}")


def main():
    """Run all examples."""
    print("\n🚀 VetGraph Multi-Provider Usage Examples")
    print("=" * 60)
    print("\nThese examples show how VetGraph works with ANY LLM provider")
    print("using the same API interface!\n")
    
    # Check which API keys are available
    available_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    }
    
    print("Available API Keys:")
    for key_name, key_value in available_keys.items():
        status = "✅" if key_value else "❌"
        print(f"  {status} {key_name}")
    
    # Run examples based on available keys
    if os.getenv("OPENAI_API_KEY"):
        try:
            example_openai()
            example_custom_client()
        except Exception as e:
            print(f"\n❌ OpenAI example failed: {e}")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            example_anthropic()
        except Exception as e:
            print(f"\n❌ Anthropic example failed: {e}")
    
    if os.getenv("GOOGLE_API_KEY"):
        try:
            example_gemini()
        except Exception as e:
            print(f"\n❌ Gemini example failed: {e}")
    
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        try:
            example_azure_openai()
        except Exception as e:
            print(f"\n❌ Azure OpenAI example failed: {e}")
    
    # Ollama doesn't need API keys
    print("\n💡 Want to try Ollama (free, local)?")
    print("   1. Install from: https://ollama.ai")
    print("   2. Run: ollama pull llama3.2")
    print("   3. Uncomment example_ollama() call in this script")
    
    # Uncomment to test Ollama:
    # try:
    #     example_ollama()
    # except Exception as e:
    #     print(f"\n❌ Ollama example failed: {e}")
    #     print("   Make sure Ollama is installed and running!")
    
    print("\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
