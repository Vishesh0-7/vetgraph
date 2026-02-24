"""Unit tests for VetGraph library."""

import pytest
from vetgraph.schema import Entity, Relationship, KnowledgeGraph


class TestEntity:
    """Test Entity model."""
    
    def test_create_entity(self):
        """Test creating a valid entity."""
        entity = Entity(
            id="person_1",
            label="Albert Einstein",
            type="Person",
            properties={"birth_year": 1879}
        )
        assert entity.id == "person_1"
        assert entity.label == "Albert Einstein"
        assert entity.type == "Person"
        assert entity.properties["birth_year"] == 1879
    
    def test_entity_validation(self):
        """Test entity validation."""
        with pytest.raises(ValueError):
            Entity(id="", label="Test", type="Person")
        
        with pytest.raises(ValueError):
            Entity(id="test_1", label="", type="Person")


class TestRelationship:
    """Test Relationship model."""
    
    def test_create_relationship(self):
        """Test creating a valid relationship."""
        rel = Relationship(
            source="person_1",
            target="org_1",
            type="works_for",
            weight=1.0
        )
        assert rel.source == "person_1"
        assert rel.target == "org_1"
        assert rel.type == "works_for"
        assert rel.weight == 1.0
    
    def test_relationship_validation(self):
        """Test relationship validation."""
        with pytest.raises(ValueError):
            Relationship(source="", target="org_1", type="works_for")


class TestKnowledgeGraph:
    """Test KnowledgeGraph model."""
    
    def test_create_knowledge_graph(self):
        """Test creating an empty knowledge graph."""
        kg = KnowledgeGraph()
        assert kg.get_entity_count() == 0
        assert kg.get_relationship_count() == 0
    
    def test_add_entity(self):
        """Test adding entities to the graph."""
        kg = KnowledgeGraph()
        entity = Entity(id="e1", label="Test", type="Concept")
        kg.add_entity(entity)
        assert kg.get_entity_count() == 1
        assert kg.get_entity("e1") == entity
    
    def test_add_relationship(self):
        """Test adding relationships to the graph."""
        kg = KnowledgeGraph()
        e1 = Entity(id="e1", label="Entity 1", type="Concept")
        e2 = Entity(id="e2", label="Entity 2", type="Concept")
        kg.add_entity(e1)
        kg.add_entity(e2)
        
        rel = Relationship(source="e1", target="e2", type="related_to")
        kg.add_relationship(rel)
        assert kg.get_relationship_count() == 1
    
    def test_add_relationship_missing_entity(self):
        """Test that adding relationship with missing entity raises error."""
        kg = KnowledgeGraph()
        e1 = Entity(id="e1", label="Entity 1", type="Concept")
        kg.add_entity(e1)
        
        rel = Relationship(source="e1", target="missing", type="related_to")
        with pytest.raises(ValueError):
            kg.add_relationship(rel)
