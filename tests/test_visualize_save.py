"""Unit tests for VetGraph visualization and save/load methods."""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch
from vetgraph import VetGraph, Triple, KnowledgeGraphExtraction


class TestVisualize:
    """Test the visualize method."""
    
    @patch('vetgraph.core.OpenAI')
    @patch('vetgraph.core.Network')
    def test_visualize_creates_html(self, mock_network_class, mock_openai):
        """Test that visualize creates an HTML file."""
        # Setup
        mock_net = Mock()
        mock_network_class.return_value = mock_net
        
        vg = VetGraph(api_key="test-key")
        vg.graph.add_edge("A", "B", relation="works_for")
        
        # Test
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            vg.visualize(output_path)
            
            # Verify Network was created with correct params
            mock_network_class.assert_called_once()
            call_kwargs = mock_network_class.call_args[1]
            assert call_kwargs['directed'] == True
            assert call_kwargs['notebook'] == False
            
            # Verify nodes and edges were added
            assert mock_net.add_node.call_count == 2  # A and B
            assert mock_net.add_edge.call_count == 1  # A -> B
            
            # Verify save was called
            mock_net.save_graph.assert_called_once_with(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('vetgraph.core.OpenAI')
    @patch('vetgraph.core.Network')
    def test_visualize_empty_graph_warning(self, mock_network_class, mock_openai, capsys):
        """Test that visualizing an empty graph shows a warning."""
        vg = VetGraph(api_key="test-key")
        
        vg.visualize('test.html')
        
        captured = capsys.readouterr()
        assert "Warning: Graph is empty" in captured.out
        
        # Should not create the network
        mock_network_class.assert_not_called()
    
    @patch('vetgraph.core.OpenAI')
    @patch('vetgraph.core.Network')
    def test_visualize_includes_relation_on_hover(self, mock_network_class, mock_openai):
        """Test that edges have relation as title (for hover)."""
        mock_net = Mock()
        mock_network_class.return_value = mock_net
        
        vg = VetGraph(api_key="test-key")
        vg.graph.add_edge("Einstein", "Princeton", relation="worked_at")
        
        vg.visualize('test.html')
        
        # Find the add_edge call and verify the relation is in title
        edge_calls = mock_net.add_edge.call_args_list
        assert len(edge_calls) == 1
        
        args, kwargs = edge_calls[0]
        assert args[0] == "Einstein"
        assert args[1] == "Princeton"
        assert kwargs['title'] == "worked_at"
        assert kwargs['label'] == "worked_at"


class TestSaveGraph:
    """Test the save_graph method."""
    
    @patch('vetgraph.core.OpenAI')
    def test_save_graph_json(self, mock_openai):
        """Test saving graph as JSON."""
        vg = VetGraph(api_key="test-key")
        vg.graph.add_edge("A", "B", relation="test_relation")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            vg.save_graph(output_path, format='json')
            
            # Verify file exists and contains valid JSON
            assert os.path.exists(output_path)
            
            import json
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            # Check structure
            assert 'nodes' in data
            assert 'links' in data or 'edges' in data
            assert len(data['nodes']) == 2
            assert len(data.get('links', data.get('edges', []))) == 1
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('vetgraph.core.OpenAI')
    def test_save_graph_graphml(self, mock_openai):
        """Test saving graph as GraphML."""
        vg = VetGraph(api_key="test-key")
        vg.graph.add_edge("A", "B", relation="test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            vg.save_graph(output_path, format='graphml')
            assert os.path.exists(output_path)
            
            # Verify it's XML
            with open(output_path, 'r') as f:
                content = f.read()
            assert '<?xml' in content or '<graphml' in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('vetgraph.core.OpenAI')
    def test_save_graph_invalid_format(self, mock_openai):
        """Test that invalid format raises ValueError."""
        vg = VetGraph(api_key="test-key")
        vg.graph.add_edge("A", "B", relation="test")
        
        with pytest.raises(ValueError, match="Unsupported format"):
            vg.save_graph('test.xyz', format='invalid_format')
    
    @patch('vetgraph.core.OpenAI')
    def test_save_graph_empty_warning(self, mock_openai, capsys):
        """Test that saving an empty graph shows a warning."""
        vg = VetGraph(api_key="test-key")
        
        vg.save_graph('test.json', format='json')
        
        captured = capsys.readouterr()
        assert "Warning: Graph is empty" in captured.out


class TestLoadGraph:
    """Test the load_graph method."""
    
    @patch('vetgraph.core.OpenAI')
    def test_load_graph_json(self, mock_openai):
        """Test loading a graph from JSON."""
        # Create and save a graph
        vg1 = VetGraph(api_key="test-key")
        vg1.graph.add_edge("Einstein", "Princeton", relation="worked_at")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            save_path = tmp.name
        
        try:
            vg1.save_graph(save_path, format='json')
            
            # Load into new instance
            vg2 = VetGraph(api_key="test-key")
            vg2.load_graph(save_path, format='json')
            
            # Verify loaded correctly
            assert vg2.graph.number_of_nodes() == 2
            assert vg2.graph.number_of_edges() == 1
            assert vg2.graph.has_edge("Einstein", "Princeton")
            
            edge_data = vg2.graph.get_edge_data("Einstein", "Princeton")
            assert edge_data['relation'] == "worked_at"
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    @patch('vetgraph.core.OpenAI')
    def test_load_graph_replaces_existing(self, mock_openai):
        """Test that loading a graph replaces the existing one."""
        # Create graph with data
        vg1 = VetGraph(api_key="test-key")
        vg1.graph.add_edge("A", "B", relation="test1")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            save_path = tmp.name
        
        try:
            vg1.save_graph(save_path, format='json')
            
            # Create another graph with different data
            vg2 = VetGraph(api_key="test-key")
            vg2.graph.add_edge("C", "D", relation="test2")
            vg2.graph.add_edge("E", "F", relation="test3")
            
            assert vg2.graph.number_of_nodes() == 4
            
            # Load - should replace
            vg2.load_graph(save_path, format='json')
            
            # Should now have the loaded graph, not the original
            assert vg2.graph.number_of_nodes() == 2
            assert vg2.graph.number_of_edges() == 1
            assert vg2.graph.has_edge("A", "B")
            assert not vg2.graph.has_edge("C", "D")
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    @patch('vetgraph.core.OpenAI')
    def test_load_graph_invalid_format(self, mock_openai):
        """Test that loading with invalid format raises ValueError."""
        vg = VetGraph(api_key="test-key")
        
        with pytest.raises(ValueError, match="Unsupported format"):
            vg.load_graph('test.xyz', format='invalid_format')
    
    @patch('vetgraph.core.OpenAI')
    def test_load_graph_file_not_found(self, mock_openai):
        """Test that loading non-existent file raises FileNotFoundError."""
        vg = VetGraph(api_key="test-key")
        
        with pytest.raises(FileNotFoundError):
            vg.load_graph('nonexistent_file.json', format='json')


class TestRoundTrip:
    """Test saving and loading graphs in different formats."""
    
    @patch('vetgraph.core.OpenAI')
    def test_round_trip_json(self, mock_openai):
        """Test save and load preserves graph structure (JSON)."""
        vg1 = VetGraph(api_key="test-key")
        vg1.graph.add_edge("A", "B", relation="rel1")
        vg1.graph.add_edge("B", "C", relation="rel2")
        vg1.graph.add_edge("A", "C", relation="rel3")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            path = tmp.name
        
        try:
            vg1.save_graph(path, format='json')
            
            vg2 = VetGraph(api_key="test-key")
            vg2.load_graph(path, format='json')
            
            assert vg2.graph.number_of_nodes() == vg1.graph.number_of_nodes()
            assert vg2.graph.number_of_edges() == vg1.graph.number_of_edges()
            
            for edge in vg1.graph.edges():
                assert vg2.graph.has_edge(*edge)
        finally:
            if os.path.exists(path):
                os.unlink(path)
