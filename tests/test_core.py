"""
Comprehensive unit tests for the VetGraph class.

This test suite mocks all LLM API calls (via instructor) so tests can run offline
without an internet connection or spending API credits.

Updated to work with the provider-agnostic refactored VetGraph using instructor.
"""

import pytest
import networkx as nx
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch, call
from vetgraph import VetGraph, Triple, KnowledgeGraphExtraction
import instructor


# ============================================================================
# FIXTURES AND HELPERS
# ============================================================================

@pytest.fixture
def mock_instructor_client():
    """Fixture that provides a fully mocked instructor client."""
    mock_client = Mock(spec=instructor.Instructor)
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    return mock_client


def create_mock_extraction(triples_data):
    """
    Helper function to create a mock extraction response.
    
    Args:
        triples_data: List of tuples (subject, predicate, object)
    
    Returns:
        KnowledgeGraphExtraction object (instructor returns the model directly)
    """
    triples = [
        Triple(subject=s, predicate=p, object=o)
        for s, p, o in triples_data
    ]
    return KnowledgeGraphExtraction(triples=triples)


@pytest.fixture
def sample_triples():
    """Sample triples for testing."""
    return [
        ("Einstein", "developed", "relativity"),
        ("Einstein", "worked_at", "Princeton"),
        ("Einstein", "won", "Nobel Prize")
    ]


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestVetGraphInit:
    """Test VetGraph initialization."""
    
    def test_init_creates_empty_digraph(self, mock_openai_client):
        """Test that initialization creates an empty directed graph."""
        vg = VetGraph(api_key="test-key")
        
        assert isinstance(vg.graph, nx.DiGraph)
        assert vg.graph.number_of_nodes() == 0
        assert vg.graph.number_of_edges() == 0
        assert vg.graph.is_directed() == True
    
    def test_init_with_client(self, mock_instructor_client):
        """Test initialization with instructor client."""
        vg = VetGraph(client=mock_instructor_client, model="gpt-4")
        assert vg.client == mock_instructor_client
        assert vg.model == "gpt-4"
        assert isinstance(vg.graph, nx.DiGraph)
    
    def test_init_default_model(self, mock_instructor_client):
        """Test default model is gpt-4o-mini."""
        vg = VetGraph(client=mock_instructor_client)
        assert vg.model == "gpt-4o-mini"
    
    @patch('vetgraph.providers.create_openai_client')
    def test_from_openai_factory(self, mock_create):
        """Test from_openai factory method."""
        mock_client = Mock(spec=instructor.Instructor)
        mock_create.return_value = mock_client
        
        vg = VetGraph.from_openai(api_key="test-key", model="gpt-4")
        
        assert vg.client == mock_client
        assert vg.model == "gpt-4"
        mock_create.assert_called_once()
    
    @patch('vetgraph.providers.create_ollama_client')
    def test_from_ollama_factory(self, mock_create):
        """Test from_ollama factory method."""
        mock_client = Mock(spec=instructor.Instructor)
        mock_create.return_value = mock_client
        
        vg = VetGraph.from_ollama(model="llama3.2")
        
        assert vg.client == mock_client
        assert vg.model == "llama3.2"
        mock_create.assert_called_once()


# ============================================================================
# PROMPT GENERATION TESTS
# ============================================================================

class TestGetExtractionPrompt:
    """Test the _get_extraction_prompt method."""
    
    def test_prompt_without_schema(self, mock_instructor_client):
        """Test prompt generation without schema constraints."""
        vg = VetGraph(client=mock_instructor_client)
        prompt = vg._get_extraction_prompt("test text")
        
        # Check for key phrases
        assert "linguistic analyst" in prompt
        assert "subject, predicate, object" in prompt or "(subject, predicate, object)" in prompt
        assert "directional relationships" in prompt
        
        # Should not have schema restrictions
        assert "MUST use only these predicates" not in prompt
    
    def test_prompt_with_schema(self, mock_instructor_client):
        """Test prompt generation with schema constraints."""
        vg = VetGraph(client=mock_instructor_client)
        schema = ["works_for", "born_in", "lives_in"]
        prompt = vg._get_extraction_prompt("test text", schema=schema)
        
        # Check that schema is enforced
        assert "linguistic analyst" in prompt
        assert "works_for" in prompt
        assert "born_in" in prompt
        assert "lives_in" in prompt
        assert "MUST" in prompt or "only these predicates" in prompt
    
    def test_prompt_with_empty_schema(self, mock_instructor_client):
        """Test prompt with empty schema list."""
        vg = VetGraph(client=mock_instructor_client)
        prompt = vg._get_extraction_prompt("test text", schema=[])
        
        # Empty schema should still generate prompt
        assert len(prompt) > 0
        assert "linguistic analyst" in prompt


# ============================================================================
# ADD_TEXT METHOD TESTS (CORE FUNCTIONALITY)
# ============================================================================

class TestAddText:
    """Test the add_text method with comprehensive mocking."""
    
    def test_add_text_increases_node_count(self, mock_instructor_client, sample_triples):
        """Test that adding triples correctly increases the node count."""
        # Setup mock to return sample triples (instructor returns model directly)
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        
        # Initially empty
        assert vg.graph.number_of_nodes() == 0
        
        # Add text
        result = vg.add_text("Einstein developed relativity and worked at Princeton.")
        
        # Verify node count increased
        # Should have: Einstein, relativity, Princeton, Nobel Prize (4 unique nodes)
        assert vg.graph.number_of_nodes() == 4
        
        # Verify specific nodes exist
        assert "Einstein" in vg.graph.nodes()
        assert "relativity" in vg.graph.nodes()
        assert "Princeton" in vg.graph.nodes()
        assert "Nobel Prize" in vg.graph.nodes()
    
    def test_add_text_increases_edge_count(self, mock_openai_client, sample_triples):
        """Test that adding triples correctly increases the edge count."""
        mock_completion = create_mock_extraction(sample_triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        
        # Initially empty
        assert vg.graph.number_of_edges() == 0
        
        # Add text
        result = vg.add_text("test text")
        
        # Should have 3 edges (one per triple)
        assert vg.graph.number_of_edges() == 3
    
    def test_add_text_creates_correct_edges(self, mock_openai_client):
        """Test that edges are created between correct nodes."""
        triples = [
            ("Einstein", "developed", "relativity"),
            ("Einstein", "worked_at", "Princeton")
        ]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test text")
        
        # Verify edges exist
        assert vg.graph.has_edge("Einstein", "relativity")
        assert vg.graph.has_edge("Einstein", "Princeton")
        
        # Verify edge count
        assert vg.graph.number_of_edges() == 2
    
    def test_add_text_stores_relation_attribute(self, mock_openai_client):
        """Test that edges store the relation (predicate) as an attribute."""
        triples = [("A", "works_for", "B")]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test")
        
        # Verify edge has relation attribute
        edge_data = vg.graph.get_edge_data("A", "B")
        assert edge_data is not None
        assert "relation" in edge_data
        assert edge_data["relation"] == "works_for"
    
    def test_add_text_returns_extraction(self, mock_openai_client, sample_triples):
        """Test that add_text returns the KnowledgeGraphExtraction object."""
        mock_completion = create_mock_extraction(sample_triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        result = vg.add_text("test text")
        
        # Verify returned object
        assert isinstance(result, KnowledgeGraphExtraction)
        assert result.get_triple_count() == 3
        assert len(result.triples) == 3
    
    def test_add_text_multiple_times_accumulates(self, mock_openai_client):
        """Test that calling add_text multiple times accumulates nodes and edges."""
        # First batch of triples
        triples1 = [("A", "rel1", "B")]
        mock_completion1 = create_mock_extraction(triples1)
        
        # Second batch of triples
        triples2 = [("C", "rel2", "D"), ("A", "rel3", "D")]
        mock_completion2 = create_mock_extraction(triples2)
        
        mock_openai_client.beta.chat.completions.parse.side_effect = [
            mock_completion1,
            mock_completion2
        ]
        
        vg = VetGraph(api_key="test-key")
        
        # First add
        vg.add_text("first text")
        assert vg.graph.number_of_nodes() == 2  # A, B
        assert vg.graph.number_of_edges() == 1
        
        # Second add - should accumulate
        vg.add_text("second text")
        assert vg.graph.number_of_nodes() == 4  # A, B, C, D
        assert vg.graph.number_of_edges() == 3  # 1 + 2
    
    def test_add_text_with_duplicate_node(self, mock_openai_client):
        """Test that duplicate nodes don't increase node count."""
        triples = [
            ("Einstein", "developed", "relativity"),
            ("Einstein", "won", "Nobel Prize")  # Einstein appears twice
        ]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test")
        
        # Should have 3 unique nodes, not 4
        assert vg.graph.number_of_nodes() == 3
        assert vg.graph.number_of_edges() == 2
    
    def test_add_text_with_schema_passes_to_prompt(self, mock_openai_client):
        """Test that schema parameter is passed to prompt generation."""
        mock_completion = create_mock_extraction([])
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        schema = ["works_for", "born_in"]
        
        # Spy on the _get_extraction_prompt method
        with patch.object(vg, '_get_extraction_prompt', wraps=vg._get_extraction_prompt) as spy:
            vg.add_text("test text", schema=schema)
            
            # Verify schema was passed
            spy.assert_called_once()
            call_args = spy.call_args
            assert call_args[1]['schema'] == schema
    
    def test_add_text_with_temperature(self, mock_openai_client):
        """Test that temperature parameter is passed to API."""
        mock_completion = create_mock_extraction([])
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test text", temperature=0.7)
        
        # Verify API was called with correct temperature
        call_kwargs = mock_openai_client.beta.chat.completions.parse.call_args[1]
        assert call_kwargs['temperature'] == 0.7
    
    def test_add_text_with_empty_extraction(self, mock_openai_client):
        """Test behavior when LLM returns no triples."""
        mock_completion = create_mock_extraction([])
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        result = vg.add_text("test text")
        
        # Graph should remain empty
        assert vg.graph.number_of_nodes() == 0
        assert vg.graph.number_of_edges() == 0
        assert result.get_triple_count() == 0
    
    def test_add_text_raises_on_none_extraction(self, mock_openai_client):
        """Test that ValueError is raised when extraction is None."""
        mock_message = Mock()
        mock_message.parsed = None
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        
        with pytest.raises(ValueError, match="Failed to parse extraction"):
            vg.add_text("test text")
    
    def test_add_text_calls_openai_api(self, mock_openai_client):
        """Test that add_text makes the correct API call."""
        mock_completion = create_mock_extraction([("A", "rel", "B")])
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test text", temperature=0.5)
        
        # Verify API was called
        mock_openai_client.beta.chat.completions.parse.assert_called_once()
        
        # Verify call parameters
        call_args = mock_openai_client.beta.chat.completions.parse.call_args
        call_kwargs = call_args[1]
        
        assert call_kwargs['model'] == "gpt-4o-mini"
        assert call_kwargs['response_format'] == KnowledgeGraphExtraction
        assert call_kwargs['temperature'] == 0.5
        assert len(call_kwargs['messages']) == 2
        assert call_kwargs['messages'][0]['role'] == 'system'
        assert call_kwargs['messages'][1]['role'] == 'user'


# ============================================================================
# GRAPH MANAGEMENT TESTS
# ============================================================================

class TestGraphManagement:
    """Test graph management methods."""
    
    def test_get_graph(self, mock_openai_client):
        """Test get_graph returns the internal graph."""
        vg = VetGraph(api_key="test-key")
        graph = vg.get_graph()
        
        assert isinstance(graph, nx.DiGraph)
        assert graph is vg.graph
    
    def test_clear_graph(self, mock_openai_client):
        """Test clear_graph removes all nodes and edges."""
        vg = VetGraph(api_key="test-key")
        vg.graph.add_edge("A", "B", relation="test")
        vg.graph.add_edge("C", "D", relation="test2")
        
        assert vg.graph.number_of_nodes() == 4
        assert vg.graph.number_of_edges() == 2
        
        vg.clear_graph()
        
        assert vg.graph.number_of_nodes() == 0
        assert vg.graph.number_of_edges() == 0
    
    def test_clear_graph_on_empty_graph(self, mock_openai_client):
        """Test clear_graph on already empty graph."""
        vg = VetGraph(api_key="test-key")
        
        vg.clear_graph()  # Should not raise error
        
        assert vg.graph.number_of_nodes() == 0
        assert vg.graph.number_of_edges() == 0
    
    def test_get_statistics_empty_graph(self, mock_openai_client):
        """Test statistics for empty graph."""
        vg = VetGraph(api_key="test-key")
        stats = vg.get_statistics()
        
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["density"] == 0.0
        assert stats["unique_relations"] == 0
        assert stats["is_connected"] == False
    
    def test_get_statistics_with_data(self, mock_openai_client):
        """Test statistics for graph with data."""
        vg = VetGraph(api_key="test-key")
        vg.graph.add_edge("A", "B", relation="works_for")
        vg.graph.add_edge("B", "C", relation="manages")
        vg.graph.add_edge("A", "C", relation="works_for")
        
        stats = vg.get_statistics()
        
        assert stats["nodes"] == 3
        assert stats["edges"] == 3
        assert stats["unique_relations"] == 2  # works_for and manages
        assert stats["density"] > 0
    
    def test_get_statistics_calculates_density(self, mock_openai_client):
        """Test that density is calculated correctly."""
        vg = VetGraph(api_key="test-key")
        
        # Add a complete directed graph: 3 nodes, all connected
        vg.graph.add_edge("A", "B", relation="r1")
        vg.graph.add_edge("A", "C", relation="r2")
        vg.graph.add_edge("B", "A", relation="r3")
        vg.graph.add_edge("B", "C", relation="r4")
        vg.graph.add_edge("C", "A", relation="r5")
        vg.graph.add_edge("C", "B", relation="r6")
        
        stats = vg.get_statistics()
        
        # Density for directed graph = edges / (nodes * (nodes - 1))
        # = 6 / (3 * 2) = 6 / 6 = 1.0
        assert stats["density"] == 1.0
    
    def test_get_statistics_counts_unique_relations(self, mock_openai_client):
        """Test that unique relations are counted correctly."""
        vg = VetGraph(api_key="test-key")
        
        vg.graph.add_edge("A", "B", relation="works_for")
        vg.graph.add_edge("C", "D", relation="works_for")  # Same relation
        vg.graph.add_edge("E", "F", relation="manages")     # Different relation
        
        stats = vg.get_statistics()
        
        # Should count 2 unique relations even though works_for appears twice
        assert stats["unique_relations"] == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_incremental_graph_building(self, mock_openai_client):
        """Test building a graph incrementally with multiple add_text calls."""
        # Mock different responses for each call
        responses = [
            create_mock_extraction([("A", "rel1", "B")]),
            create_mock_extraction([("B", "rel2", "C")]),
            create_mock_extraction([("C", "rel3", "A")])
        ]
        mock_openai_client.beta.chat.completions.parse.side_effect = responses
        
        vg = VetGraph(api_key="test-key")
        
        # Add text incrementally
        vg.add_text("First text")
        assert vg.graph.number_of_nodes() == 2
        assert vg.graph.number_of_edges() == 1
        
        vg.add_text("Second text")
        assert vg.graph.number_of_nodes() == 3
        assert vg.graph.number_of_edges() == 2
        
        vg.add_text("Third text")
        assert vg.graph.number_of_nodes() == 3  # No new nodes
        assert vg.graph.number_of_edges() == 3
    
    def test_complex_knowledge_graph(self, mock_openai_client):
        """Test building a complex knowledge graph."""
        triples = [
            ("Einstein", "born_in", "Germany"),
            ("Einstein", "moved_to", "USA"),
            ("Einstein", "worked_at", "Princeton"),
            ("Princeton", "located_in", "New Jersey"),
            ("New Jersey", "part_of", "USA"),
            ("Einstein", "developed", "relativity"),
            ("Einstein", "won", "Nobel Prize")
        ]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("Complex biography of Einstein")
        
        # Verify graph structure
        assert vg.graph.number_of_nodes() == 7
        assert vg.graph.number_of_edges() == 7
        
        # Verify specific paths exist
        assert vg.graph.has_edge("Einstein", "Germany")
        assert vg.graph.has_edge("Princeton", "New Jersey")
        assert vg.graph.has_edge("New Jersey", "USA")
        
        # Verify statistics
        stats = vg.get_statistics()
        assert stats["unique_relations"] == 7
    
    def test_workflow_add_visualize_save(self, mock_openai_client):
        """Test complete workflow: add text, get stats, clear."""
        triples = [("A", "rel", "B")]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        
        # Add text
        extraction = vg.add_text("test text")
        assert extraction.get_triple_count() == 1
        
        # Get statistics
        stats = vg.get_statistics()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        
        # Get graph
        graph = vg.get_graph()
        assert graph.number_of_nodes() == 2
        
        # Clear
        vg.clear_graph()
        assert vg.graph.number_of_nodes() == 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_self_loop(self, mock_openai_client):
        """Test handling of self-loops (entity relating to itself)."""
        triples = [("Python", "is_a", "Python")]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test")
        
        # Should create only 1 node but still have an edge
        assert vg.graph.number_of_nodes() == 1
        assert vg.graph.number_of_edges() == 1
        assert vg.graph.has_edge("Python", "Python")
    
    def test_parallel_edges_same_relation(self, mock_openai_client):
        """Test parallel edges with the same relation (should overwrite in DiGraph)."""
        # Add same edge multiple times
        responses = [
            create_mock_extraction([("A", "rel", "B")]),
            create_mock_extraction([("A", "rel", "B")])
        ]
        mock_openai_client.beta.chat.completions.parse.side_effect = responses
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("first")
        vg.add_text("second")
        
        # DiGraph allows only one edge between two nodes
        assert vg.graph.number_of_edges() == 1
    
    def test_parallel_edges_different_relations(self, mock_openai_client):
        """Test parallel edges with different relations (last one wins)."""
        triples = [
            ("A", "rel1", "B"),
            ("A", "rel2", "B")  # Different relation, same nodes
        ]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test")
        
        # Only one edge between A and B (last one wins)
        assert vg.graph.number_of_edges() == 1
        edge_data = vg.graph.get_edge_data("A", "B")
        assert edge_data["relation"] == "rel2"  # Last one should win
    
    def test_bidirectional_edges(self, mock_openai_client):
        """Test bidirectional edges between same nodes."""
        triples = [
            ("A", "rel1", "B"),
            ("B", "rel2", "A")
        ]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test")
        
        # Should have 2 edges (one in each direction)
        assert vg.graph.number_of_edges() == 2
        assert vg.graph.has_edge("A", "B")
        assert vg.graph.has_edge("B", "A")
    
    def test_very_long_entity_names(self, mock_openai_client):
        """Test handling of very long entity names."""
        long_name = "A" * 1000
        triples = [(long_name, "rel", "B")]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test")
        
        assert vg.graph.number_of_nodes() == 2
        assert long_name in vg.graph.nodes()
    
    def test_special_characters_in_entities(self, mock_openai_client):
        """Test handling of special characters in entity names."""
        triples = [
            ("Entity-1", "rel", "Entity_2"),
            ("Entity@3", "rel", "Entity#4"),
            ("Entity 5", "rel", "Entity.6")
        ]
        mock_completion = create_mock_extraction(triples)
        mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
        
        vg = VetGraph(api_key="test-key")
        vg.add_text("test")
        
        # Should handle all special characters
        assert vg.graph.number_of_nodes() == 6
        assert vg.graph.number_of_edges() == 3


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
