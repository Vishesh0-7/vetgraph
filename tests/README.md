# VetGraph Test Suite

This directory contains comprehensive tests for the VetGraph library. All tests use mocked OpenAI API responses, allowing you to run them **offline without an internet connection or spending API credits**.

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_core.py -v
pytest tests/test_schema.py -v
pytest tests/test_triples.py -v
pytest tests/test_visualize_save.py -v
```

### Run specific test class
```bash
pytest tests/test_core.py::TestAddText -v
pytest tests/test_core.py::TestGraphManagement -v
```

### Run specific test
```bash
pytest tests/test_core.py::TestAddText::test_add_text_increases_node_count -v
```

### Run with coverage
```bash
pytest tests/ --cov=vetgraph --cov-report=html
```

## Test Organization

### test_core.py
Comprehensive tests for the main VetGraph class:
- **TestVetGraphInit**: Initialization tests
- **TestGetExtractionPrompt**: Prompt generation tests
- **TestAddText**: Core functionality (adding text and building graphs)
- **TestGraphManagement**: Graph management methods (clear, statistics)
- **TestIntegration**: Integration tests for complete workflows
- **TestEdgeCases**: Edge cases (self-loops, parallel edges, etc.)

### test_schema.py
Tests for Pydantic data models:
- **TestEntity**: Entity model validation
- **TestRelationship**: Relationship model validation
- **TestKnowledgeGraph**: KnowledgeGraph model functionality

### test_triples.py
Tests for Triple-based extraction models:
- **TestTriple**: Triple model validation
- **TestKnowledgeGraphExtraction**: Extraction container functionality

### test_visualize_save.py
Tests for visualization and I/O:
- **TestVisualize**: Interactive HTML visualization
- **TestSaveGraph**: Graph export functionality
- **TestLoadGraph**: Graph import functionality
- **TestRoundTrip**: Save/load round-trip tests

## Key Features

### 🚀 No API Calls
All tests use mocked OpenAI responses. You can run the entire test suite without:
- Internet connection
- OpenAI API key
- Spending any API credits

### 📊 Comprehensive Coverage
Tests cover:
- Node and edge counting
- Incremental graph building
- Schema-constrained extraction
- Error handling
- Edge cases (self-loops, parallel edges, special characters)
- Complete workflows

### 🎯 Focused Test Cases
Each test focuses on a single behavior with clear assertions:
```python
def test_add_text_increases_node_count(self, mock_openai_client, sample_triples):
    """Test that adding triples correctly increases the node count."""
    mock_completion = create_mock_extraction(sample_triples)
    mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
    
    vg = VetGraph(api_key="test-key")
    assert vg.graph.number_of_nodes() == 0
    
    vg.add_text("test text")
    
    assert vg.graph.number_of_nodes() == 4  # ✓ Clear assertion
```

## Mock Helpers

### create_mock_extraction(triples_data)
Helper function to create mock LLM responses:
```python
# Create mock with specific triples
triples = [
    ("Einstein", "developed", "relativity"),
    ("Einstein", "worked_at", "Princeton")
]
mock_completion = create_mock_extraction(triples)
```

### Fixtures
- `mock_openai_client`: Provides fully mocked OpenAI client
- `sample_triples`: Sample triples for testing

## Example Test Run

```bash
$ pytest tests/test_core.py -v

tests/test_core.py::TestVetGraphInit::test_init_creates_empty_digraph PASSED
tests/test_core.py::TestVetGraphInit::test_init_sets_model PASSED
tests/test_core.py::TestAddText::test_add_text_increases_node_count PASSED
tests/test_core.py::TestAddText::test_add_text_increases_edge_count PASSED
tests/test_core.py::TestAddText::test_add_text_creates_correct_edges PASSED
tests/test_core.py::TestAddText::test_add_text_multiple_times_accumulates PASSED
tests/test_core.py::TestIntegration::test_incremental_graph_building PASSED
tests/test_core.py::TestEdgeCases::test_self_loop PASSED

========================= 48 passed in 2.34s =========================
```

## Writing New Tests

### Template
```python
def test_your_feature(self, mock_openai_client):
    """Test description."""
    # Setup mock response
    triples = [("Subject", "predicate", "Object")]
    mock_completion = create_mock_extraction(triples)
    mock_openai_client.beta.chat.completions.parse.return_value = mock_completion
    
    # Create VetGraph and perform action
    vg = VetGraph(api_key="test-key")
    vg.add_text("test text")
    
    # Assert expected behavior
    assert vg.graph.number_of_nodes() == 2
```

## CI/CD Integration

Tests are designed to run in CI/CD pipelines:
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest tests/ -v --cov=vetgraph
```

## Test Data

All test data is inline or generated dynamically. No external files required.

## Debugging Tests

### Run with detailed output
```bash
pytest tests/test_core.py -vv -s
```

### Run with pdb on failure
```bash
pytest tests/test_core.py --pdb
```

### Run only failed tests
```bash
pytest tests/ --lf
```

## Performance

The test suite runs quickly since no actual API calls are made:
- Full suite: ~2-3 seconds
- Single file: <1 second

## Contributing

When adding new features:
1. Add corresponding tests
2. Use the mock helper functions
3. Ensure tests run offline
4. Add docstrings explaining what's being tested
