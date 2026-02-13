"""
Provider factory methods for creating instructor-patched LLM clients.

This module provides convenient factory methods to create instructor-patched clients
for various LLM providers (OpenAI, Anthropic, Gemini, Ollama). These patched clients
add structured output capabilities using Pydantic models.

Usage:
    from vetgraph.providers import create_openai_client, create_anthropic_client
    
    # Create an OpenAI client
    client = create_openai_client(api_key="sk-...")
    
    # Create an Anthropic client
    client = create_anthropic_client(api_key="sk-ant-...")
"""

from typing import Optional, Any
import instructor


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> instructor.Instructor:
    """
    Create an instructor-patched OpenAI client.
    
    Args:
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        base_url: Optional base URL for OpenAI API
        **kwargs: Additional arguments passed to OpenAI client
    
    Returns:
        Instructor-patched OpenAI client ready for structured outputs
    
    Example:
        >>> client = create_openai_client(api_key="sk-...")
        >>> # Use with VetGraph
        >>> vg = VetGraph(client=client, model="gpt-4o-mini")
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI is not installed. Install it with: pip install 'vetgraph[openai]'"
        )
    
    openai_client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return instructor.from_openai(openai_client)


def create_anthropic_client(
    api_key: Optional[str] = None,
    **kwargs: Any
) -> instructor.Instructor:
    """
    Create an instructor-patched Anthropic client.
    
    Args:
        api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
        **kwargs: Additional arguments passed to Anthropic client
    
    Returns:
        Instructor-patched Anthropic client ready for structured outputs
    
    Example:
        >>> client = create_anthropic_client(api_key="sk-ant-...")
        >>> # Use with VetGraph
        >>> vg = VetGraph(client=client, model="claude-3-5-sonnet-20241022")
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Anthropic is not installed. Install it with: pip install 'vetgraph[anthropic]'"
        )
    
    anthropic_client = Anthropic(api_key=api_key, **kwargs)
    return instructor.from_anthropic(anthropic_client)


def create_gemini_client(
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    **kwargs: Any
) -> instructor.Instructor:
    """
    Create an instructor-patched Google Gemini client.
    
    Args:
        api_key: Google API key (if None, uses GOOGLE_API_KEY env var)
        model: Gemini model name (default: gemini-2.5-flash)
               Valid models: gemini-3-pro, gemini-3-flash, gemini-2.5-pro, 
               gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash
        temperature: Temperature for generation (0.0-2.0, default: 0.3)
        **kwargs: Additional arguments passed to GenerationConfig
    
    Returns:
        Instructor-patched Gemini client ready for structured outputs
    
    Example:
        >>> client = create_gemini_client(api_key="...", model="gemini-2.5-flash")
        >>> # Use with VetGraph
        >>> vg = VetGraph(client=client, model=None)  # model is already set in client
    
    Note:
        For Gemini, the model and temperature MUST be set when creating the client.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "Google GenAI is not installed. Install it with: pip install 'vetgraph[gemini]'"
        )
    
    # Create the Gemini client
    client = genai.Client(api_key=api_key)
    
    # Configure generation settings
    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        **kwargs
    )
    
    # Return instructor-patched client with model and config
    return instructor.from_genai(
        client=client,
        model=model,
        generation_config=generation_config
    )


def create_ollama_client(
    base_url: str = "http://localhost:11434/v1",
    **kwargs: Any
) -> instructor.Instructor:
    """
    Create an instructor-patched Ollama client.
    
    Ollama provides an OpenAI-compatible API, so this uses the OpenAI client
    with a custom base URL pointing to your local Ollama instance.
    
    Args:
        base_url: Ollama API base URL (default: http://localhost:11434/v1)
        **kwargs: Additional arguments passed to OpenAI client
    
    Returns:
        Instructor-patched Ollama client ready for structured outputs
    
    Example:
        >>> client = create_ollama_client()
        >>> # Use with VetGraph
        >>> vg = VetGraph(client=client, model="llama3.2")
    
    Note:
        Requires Ollama to be installed and running locally.
        Download from: https://ollama.ai
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI is not installed. Install it with: pip install 'vetgraph[ollama]'"
        )
    
    ollama_client = OpenAI(
        base_url=base_url,
        api_key="ollama",  # Ollama doesn't require a real API key
        **kwargs
    )
    return instructor.from_openai(ollama_client)


def create_azure_openai_client(
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: str = "2024-02-01",
    **kwargs: Any
) -> instructor.Instructor:
    """
    Create an instructor-patched Azure OpenAI client.
    
    Args:
        api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        **kwargs: Additional arguments passed to AzureOpenAI client
    
    Returns:
        Instructor-patched Azure OpenAI client ready for structured outputs
    
    Example:
        >>> client = create_azure_openai_client(
        ...     api_key="...",
        ...     azure_endpoint="https://your-resource.openai.azure.com/"
        ... )
        >>> # Use with VetGraph
        >>> vg = VetGraph(client=client, model="gpt-4-deployment-name")
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI is not installed. Install it with: pip install 'vetgraph[openai]'"
        )
    
    azure_client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        **kwargs
    )
    return instructor.from_openai(azure_client)


__all__ = [
    "create_openai_client",
    "create_anthropic_client",
    "create_gemini_client",
    "create_ollama_client",
    "create_azure_openai_client",
]
