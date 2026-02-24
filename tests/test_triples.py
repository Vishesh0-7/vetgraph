"""Unit tests for Triple and KnowledgeGraphExtraction models."""

import pytest
from vetgraph.schema import Triple, KnowledgeGraphExtraction


class TestTriple:
    """Test Triple model for knowledge graph extraction."""
    
    def test_create_triple(self):
        """Test creating a valid triple."""
        triple = Triple(
            subject="Albert Einstein",
            predicate="developed",
            object="theory of relativity"
        )
        assert triple.subject == "Albert Einstein"
        assert triple.predicate == "developed"
        assert triple.object == "theory of relativity"
    
    def test_triple_validation_empty_subject(self):
        """Test that empty subject raises error."""
        with pytest.raises(ValueError, match="Triple components cannot be empty"):
            Triple(subject="", predicate="works_for", object="Organization")
    
    def test_triple_validation_empty_predicate(self):
        """Test that empty predicate raises error."""
        with pytest.raises(ValueError, match="Triple components cannot be empty"):
            Triple(subject="Person", predicate="", object="Location")
    
    def test_triple_validation_empty_object(self):
        """Test that empty object raises error."""
        with pytest.raises(ValueError, match="Triple components cannot be empty"):
            Triple(subject="Person", predicate="lives_in", object="")
    
    def test_triple_whitespace_stripped(self):
        """Test that whitespace is stripped from triple components."""
        triple = Triple(
            subject="  Einstein  ",
            predicate="  won  ",
            object="  Nobel Prize  "
        )
        assert triple.subject == "Einstein"
        assert triple.predicate == "won"
        assert triple.object == "Nobel Prize"


class TestKnowledgeGraphExtraction:
    """Test KnowledgeGraphExtraction model."""
    
    def test_create_empty_extraction(self):
        """Test creating an empty extraction."""
        extraction = KnowledgeGraphExtraction()
        assert extraction.get_triple_count() == 0
        assert extraction.triples == []
    
    def test_create_extraction_with_triples(self):
        """Test creating extraction with initial triples."""
        triples = [
            Triple(subject="Einstein", predicate="worked_at", object="Princeton"),
            Triple(subject="Einstein", predicate="won", object="Nobel Prize")
        ]
        extraction = KnowledgeGraphExtraction(triples=triples)
        assert extraction.get_triple_count() == 2
    
    def test_add_triple(self):
        """Test adding a triple to extraction."""
        extraction = KnowledgeGraphExtraction()
        triple = Triple(
            subject="Newton",
            predicate="discovered",
            object="gravity"
        )
        extraction.add_triple(triple)
        assert extraction.get_triple_count() == 1
        assert extraction.triples[0] == triple
    
    def test_get_subjects(self):
        """Test getting unique subjects from extraction."""
        extraction = KnowledgeGraphExtraction(
            triples=[
                Triple(subject="Einstein", predicate="worked_at", object="Princeton"),
                Triple(subject="Einstein", predicate="won", object="Nobel Prize"),
                Triple(subject="Newton", predicate="discovered", object="gravity")
            ]
        )
        subjects = extraction.get_subjects()
        assert set(subjects) == {"Einstein", "Newton"}
    
    def test_get_predicates(self):
        """Test getting unique predicates from extraction."""
        extraction = KnowledgeGraphExtraction(
            triples=[
                Triple(subject="Einstein", predicate="worked_at", object="Princeton"),
                Triple(subject="Newton", predicate="worked_at", object="Cambridge"),
                Triple(subject="Einstein", predicate="won", object="Nobel Prize")
            ]
        )
        predicates = extraction.get_predicates()
        assert set(predicates) == {"worked_at", "won"}
    
    def test_get_objects(self):
        """Test getting unique objects from extraction."""
        extraction = KnowledgeGraphExtraction(
            triples=[
                Triple(subject="Einstein", predicate="worked_at", object="Princeton"),
                Triple(subject="Newton", predicate="worked_at", object="Cambridge"),
                Triple(subject="Curie", predicate="worked_at", object="Paris")
            ]
        )
        objects = extraction.get_objects()
        assert set(objects) == {"Princeton", "Cambridge", "Paris"}
    
    def test_extraction_from_llm_response(self):
        """Test creating extraction from simulated LLM response."""
        # Simulates what would come from OpenAI structured output
        llm_response = {
            "triples": [
                {
                    "subject": "Marie Curie",
                    "predicate": "discovered",
                    "object": "radium"
                },
                {
                    "subject": "Marie Curie",
                    "predicate": "won",
                    "object": "Nobel Prize"
                }
            ]
        }
        extraction = KnowledgeGraphExtraction(**llm_response)
        assert extraction.get_triple_count() == 2
        assert extraction.triples[0].subject == "Marie Curie"
        assert extraction.triples[0].predicate == "discovered"
