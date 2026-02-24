"""
Updated unit tests for the refactored provider-agnostic VetGraph class.

This test suite mocks all LLM API calls (via instructor) so tests can run offline
without an internet connection or spending API credits.

Tests work with the new instructor-based architecture.
"""

import pytest
import networkx as nx
import os
import tempfile
from unittest.mock import Mock, patch
from vetgraph import VetGraph, Triple, KnowledgeGraphExtraction, create_openai_client
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
        ("Einstein", "won", "Nobel Prize"),
        ("relativity", "is_a", "physics theory")
    ]


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestVetGraphInit:
    """Test VetGraph initialization."""
    
    def test_init_with_client(self, mock_instructor_client):
        """Test initialization with instructor client."""
        vg = VetGraph(client=mock_instructor_client, model="gpt-4")
        assert vg.client == mock_instructor_client
        assert vg.model == "gpt-4"
        assert isinstance(vg.graph, nx.DiGraph)
        assert vg.graph.number_of_nodes() == 0
    
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
    
    @patch('vetgraph.providers.create_anthropic_client')
    def test_from_anthropic_factory(self, mock_create):
        """Test from_anthropic factory method."""
        mock_client = Mock(spec=instructor.Instructor)
        mock_create.return_value = mock_client
        
        vg = VetGraph.from_anthropic(api_key="test-key", model="claude-3-5-sonnet-20241022")
        
        assert vg.client == mock_client
        assert vg.model == "claude-3-5-sonnet-20241022"
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
# ADD_TEXT METHOD TESTS (CORE FUNCTIONALITY)
# ============================================================================

class TestAddText:
    """Test the add_text method with comprehensive mocking."""
    
    def test_add_text_increases_node_count(self, mock_instructor_client, sample_triples):
        """Test that adding triples correctly increases the node count."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        
        # Initially empty
        assert vg.graph.number_of_nodes() == 0
        
        # Add text
        result = vg.add_text("Einstein developed relativity and worked at Princeton.")
        
        # Verify node count increased
        # Should have unique nodes: Einstein, relativity, Princeton, Nobel Prize, physics theory
        assert vg.graph.number_of_nodes() == 5
        
        # Verify specific nodes exist
        assert "Einstein" in vg.graph.nodes()
        assert "relativity" in vg.graph.nodes()
        assert "Princeton" in vg.graph.nodes()
        assert "Nobel Prize" in vg.graph.nodes()
    
    def test_add_text_increases_edge_count(self, mock_instructor_client, sample_triples):
        """Test that adding triples correctly increases the edge count."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        
        # Initially empty
        assert vg.graph.number_of_edges() == 0
        
        # Add text
        result = vg.add_text("Some text")
        
        # Should have edges for each triple
        assert vg.graph.number_of_edges() == 4
    
    def test_add_text_creates_correct_edges(self, mock_instructor_client, sample_triples):
        """Test that edges are created with correct relations."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        vg.add_text("Some text")
        
        # Check specific edge
        assert vg.graph.has_edge("Einstein", "relativity")
        edge_data = vg.graph.get_edge_data("Einstein", "relativity")
        assert edge_data['relation'] == 'developed'
    
    def test_add_text_returns_extraction(self, mock_instructor_client, sample_triples):
        """Test that add_text returns KnowledgeGraphExtraction."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        result = vg.add_text("Some text")
        
        assert isinstance(result, KnowledgeGraphExtraction)
        assert result.get_triple_count() == 4
    
    def test_add_text_with_schema(self, mock_instructor_client):
        """Test add_text with schema constraint."""
        schema_triples = [
            ("Alice", "works_for", "Google"),
            ("Alice", "lives_in", "California")
        ]
        mock_extraction = create_mock_extraction(schema_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        schema = ["works_for", "lives_in", "born_in"]
        
        result = vg.add_text("Alice works for Google", schema=schema)
        
        # Verify the call was made with correct parameters
        mock_instructor_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_instructor_client.chat.completions.create.call_args[1]
        assert call_kwargs['response_model'] == KnowledgeGraphExtraction
        
        # Check that all predicates are from schema
        for triple in result.triples:
            assert triple.predicate in schema
    
    def test_add_text_multiple_times_accumulates(self, mock_instructor_client):
        """Test that multiple add_text calls accumulate in the graph."""
        # First call
        triples1 = [("Alice", "works_for", "Google")]
        mock_extraction1 = create_mock_extraction(triples1)
        
        # Second call
        triples2 = [("Bob", "works_for", "Microsoft")]
        mock_extraction2 = create_mock_extraction(triples2)
        
        mock_instructor_client.chat.completions.create.side_effect = [
            mock_extraction1,
            mock_extraction2
        ]
        
        vg = VetGraph(client=mock_instructor_client)
        
        vg.add_text("Alice works for Google")
        assert vg.graph.number_of_nodes() == 2  # Alice, Google
        
        vg.add_text("Bob works for Microsoft")
        assert vg.graph.number_of_nodes() == 4  # Alice, Google, Bob, Microsoft
        assert vg.graph.number_of_edges() == 2


# ============================================================================
# GRAPH MANAGEMENT TESTS
# ============================================================================

class TestGraphManagement:
    """Test graph management methods."""
    
    def test_get_graph(self, mock_instructor_client):
        """Test get_graph returns the NetworkX graph."""
        vg = VetGraph(client=mock_instructor_client)
        graph = vg.get_graph()
        
        assert isinstance(graph, nx.DiGraph)
        assert graph is vg.graph
    
    def test_clear_graph(self, mock_instructor_client, sample_triples):
        """Test clear_graph removes all nodes and edges."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        vg.add_text("Some text")
        
        assert vg.graph.number_of_nodes() > 0
        
        vg.clear_graph()
        
        assert vg.graph.number_of_nodes() == 0
        assert vg.graph.number_of_edges() == 0
    
    def test_get_statistics(self, mock_instructor_client, sample_triples):
        """Test get_statistics returns correct metrics."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        vg.add_text("Some text")
        
        stats = vg.get_statistics()
        
        assert 'nodes' in stats
        assert 'edges' in stats
        assert 'density' in stats
        assert stats['nodes'] == 5
        assert stats['edges'] == 4
        assert isinstance(stats['density'], float)


# ============================================================================
# VISUALIZATION TESTS
# ============================================================================

class TestVisualization:
    """Test visualization methods."""
    
    def test_visualize_creates_file(self, mock_instructor_client, sample_triples):
        """Test visualize creates an HTML file."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        vg.add_text("Some text")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_graph.html")
            vg.visualize(output_path=output_path)
            
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
    
    def test_save_graph_json(self, mock_instructor_client, sample_triples):
        """Test save_graph creates JSON file."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        vg = VetGraph(client=mock_instructor_client)
        vg.add_text("Some text")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_graph.json")
            vg.save_graph(output_path=output_path, format="json")
            
            assert os.path.exists(output_path)
            
            # Verify it's valid JSON
            import json
            with open(output_path, 'r') as f:
                data = json.load(f)
                assert 'nodes' in data or 'links' in data


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self, mock_instructor_client, sample_triples):
        """Test a complete workflow from text to visualization."""
        mock_extraction = create_mock_extraction(sample_triples)
        mock_instructor_client.chat.completions.create.return_value = mock_extraction
        
        # Initialize
        vg = VetGraph(client=mock_instructor_client)
        
        # Add text
        result = vg.add_text("Einstein developed relativity")
        assert result.get_triple_count() > 0
        
        # Check graph
        assert vg.graph.number_of_nodes() > 0
        assert vg.graph.number_of_edges() > 0
        
        # Get statistics
        stats = vg.get_statistics()
        assert stats['nodes'] > 0
        
        # Save graph
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "graph.json")
            vg.save_graph(json_path, format="json")
            assert os.path.exists(json_path)
            
            # Visualize
            html_path = os.path.join(tmpdir, "graph.html")
            vg.visualize(html_path)
            assert os.path.exists(html_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
