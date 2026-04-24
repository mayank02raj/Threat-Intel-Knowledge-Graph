"""Custom exception hierarchy for the Threat Intel Knowledge Graph."""

from __future__ import annotations


class TIKGError(Exception):
    """Base exception for all threat intel knowledge graph errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(TIKGError):
    """Raised when feed ingestion fails."""

    def __init__(self, feed_name: str, reason: str) -> None:
        super().__init__(f"Ingestion from '{feed_name}' failed: {reason}", {"feed": feed_name})
        self.feed_name = feed_name


class GraphError(TIKGError):
    """Raised when Neo4j graph operations fail."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(f"Graph operation '{operation}' failed: {reason}")
        self.operation = operation


class PredictionError(TIKGError):
    """Raised when GNN prediction fails."""

    def __init__(self, actor_id: str, reason: str) -> None:
        super().__init__(f"Prediction for actor '{actor_id}' failed: {reason}")
        self.actor_id = actor_id


class AnalysisError(TIKGError):
    """Raised when temporal trend analysis fails."""

    def __init__(self, technique_id: str, reason: str) -> None:
        super().__init__(f"Analysis for technique '{technique_id}' failed: {reason}")
        self.technique_id = technique_id
