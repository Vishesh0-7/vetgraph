"""
Pydantic models for knowledge graph entities and relationships.

This module defines the data schemas used throughout VetGraph for representing
entities, relationships, and knowledge graphs with strong type validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class Triple(BaseModel):
    """
    Represents a single knowledge triple (subject-predicate-object).
    
    This model is designed for structured outputs from LLMs via the OpenAI SDK.
    Each triple represents a single fact or relationship extracted from text,
    following the RDF triple pattern.
    
    Attributes:
        subject: The entity or concept the triple is about
        predicate: The relationship or property connecting subject and object
        object: The entity, value, or concept related to the subject
        
    Example:
        Triple(
            subject="Albert Einstein",
            predicate="developed",
            object="theory of relativity"
        )
    """
    subject: str = Field(..., description="The subject entity of the triple")
    predicate: str = Field(..., description="The relationship or action connecting subject and object")
    object: str = Field(..., description="The object entity or value of the triple")
    
    @field_validator('subject', 'predicate', 'object')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure all triple components are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Triple components cannot be empty")
        return v.strip()


class KnowledgeGraphExtraction(BaseModel):
    """
    Container for knowledge graph triples extracted by an LLM.
    
    This model is specifically designed to work with OpenAI's structured output
    feature (response_format with json_schema). It enforces that the LLM returns
    a well-structured list of knowledge triples that can be validated and processed.
    
    The structured output ensures type safety and validation at the API level,
    making LLM responses more reliable and reducing parsing errors.
    
    Attributes:
        triples: List of extracted knowledge triples from the input text
        
    Example:
        extraction = KnowledgeGraphExtraction(
            triples=[
                Triple(subject="Einstein", predicate="worked_at", object="Princeton"),
                Triple(subject="Einstein", predicate="won", object="Nobel Prize")
            ]
        )
    """
    triples: List[Triple] = Field(
        default_factory=list, 
        description="List of knowledge triples extracted from text"
    )
    
    def get_triple_count(self) -> int:
        """Get the number of triples in the extraction."""
        return len(self.triples)
    
    def add_triple(self, triple: Triple) -> None:
        """Add a triple to the extraction."""
        self.triples.append(triple)
    
    def get_subjects(self) -> List[str]:
        """Get all unique subjects from the triples."""
        return list(set(triple.subject for triple in self.triples))
    
    def get_predicates(self) -> List[str]:
        """Get all unique predicates from the triples."""
        return list(set(triple.predicate for triple in self.triples))
    
    def get_objects(self) -> List[str]:
        """Get all unique objects from the triples."""
        return list(set(triple.object for triple in self.triples))


class Entity(BaseModel):
    """
    Represents a node/entity in the knowledge graph.
    
    Attributes:
        id: Unique identifier for the entity
        label: Human-readable label for the entity
        type: Category or type of the entity (e.g., "Person", "Organization")
        properties: Additional metadata about the entity
    """
    id: str = Field(..., description="Unique identifier for the entity")
    label: str = Field(..., description="Human-readable label")
    type: str = Field(..., description="Entity type or category")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Entity ID cannot be empty")
        return v.strip()
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Ensure label is not empty."""
        if not v or not v.strip():
            raise ValueError("Entity label cannot be empty")
        return v.strip()


class Relationship(BaseModel):
    """
    Represents an edge/relationship between two entities in the knowledge graph.
    
    Attributes:
        source: ID of the source entity
        target: ID of the target entity
        type: Type of relationship (e.g., "works_for", "located_in")
        properties: Additional metadata about the relationship
        weight: Optional weight/strength of the relationship
    """
    source: str = Field(..., description="Source entity ID")
    target: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    weight: float = Field(default=1.0, ge=0.0, description="Relationship weight")
    
    @field_validator('source', 'target')
    @classmethod
    def validate_entity_refs(cls, v: str) -> str:
        """Ensure entity references are not empty."""
        if not v or not v.strip():
            raise ValueError("Entity reference cannot be empty")
        return v.strip()


class KnowledgeGraph(BaseModel):
    """
    Complete knowledge graph representation with entities and relationships.
    
    Attributes:
        entities: List of entities in the graph
        relationships: List of relationships between entities
        metadata: Additional information about the graph
    """
    entities: List[Entity] = Field(default_factory=list, description="Graph entities/nodes")
    relationships: List[Relationship] = Field(default_factory=list, description="Graph edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by its ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        if not self.get_entity(entity.id):
            self.entities.append(entity)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        # Verify that both entities exist
        if not self.get_entity(relationship.source):
            raise ValueError(f"Source entity '{relationship.source}' not found in graph")
        if not self.get_entity(relationship.target):
            raise ValueError(f"Target entity '{relationship.target}' not found in graph")
        self.relationships.append(relationship)
    
    def get_entity_count(self) -> int:
        """Get the number of entities in the graph."""
        return len(self.entities)
    
    def get_relationship_count(self) -> int:
        """Get the number of relationships in the graph."""
        return len(self.relationships)


class ExtractionPrompt(BaseModel):
    """
    Configuration for LLM-based entity and relationship extraction.
    
    Attributes:
        system_prompt: System-level instructions for the LLM
        user_template: Template for user messages with text to analyze
        entity_types: List of entity types to extract
        relationship_types: List of relationship types to extract
    """
    system_prompt: str = Field(
        default=(
            "You are an expert at extracting entities and relationships from text to build "
            "knowledge graphs. Extract entities with their types and relationships between them."
        ),
        description="System instructions for the LLM"
    )
    user_template: str = Field(
        default="Extract entities and relationships from the following text:\n\n{text}",
        description="Template for user prompt"
    )
    entity_types: List[str] = Field(
        default_factory=lambda: ["Person", "Organization", "Location", "Concept", "Event"],
        description="Entity types to extract"
    )
    relationship_types: List[str] = Field(
        default_factory=lambda: ["related_to", "works_for", "located_in", "created_by", "part_of"],
        description="Relationship types to extract"
    )
