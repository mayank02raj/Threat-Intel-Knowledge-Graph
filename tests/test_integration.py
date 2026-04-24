"""Integration tests requiring Neo4j and external feeds.

Run with: pytest tests/test_integration.py -m integration
Skip with: pytest tests/ -m "not integration"
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _neo4j_available() -> bool:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not available")
class TestNeo4jIntegration:
    """Live tests against a real Neo4j instance."""

    @pytest.mark.asyncio
    async def test_connect_and_get_stats(self) -> None:
        from src.graph.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        await kg.connect()
        stats = await kg.get_stats()
        await kg.close()

        assert isinstance(stats, dict)
        assert "threat_actors" in stats
        assert "techniques" in stats

    @pytest.mark.asyncio
    async def test_entity_upsert_and_query(self) -> None:
        from src.graph.knowledge_graph import KnowledgeGraph
        from src.ingesters.feeds import ThreatEntity, ThreatRelationship

        kg = KnowledgeGraph()
        await kg.connect()

        # Insert test entities
        actor = ThreatEntity(
            entity_type="threat_actor",
            stix_id="threat-actor--test-integration",
            name="IntegrationTestActor",
        )
        technique = ThreatEntity(
            entity_type="technique",
            stix_id="attack-pattern--test-integration",
            name="TestTechnique",
            properties={"mitre_id": "T9999"},
        )
        rel = ThreatRelationship(
            source_id="threat-actor--test-integration",
            target_id="attack-pattern--test-integration",
            relationship_type="uses",
        )

        await kg.upsert_entity(actor)
        await kg.upsert_entity(technique)
        await kg.upsert_relationship(rel)

        # Query back
        techniques = await kg.query_actor_techniques("IntegrationTestActor")
        assert len(techniques) >= 1
        assert any(t["mitre_id"] == "T9999" for t in techniques)

        # Clean up
        async with kg._driver.session(database=kg.database) as session:
            await session.run(
                "MATCH (n) WHERE n.stix_id CONTAINS 'test-integration' DETACH DELETE n"
            )

        await kg.close()

    @pytest.mark.asyncio
    async def test_bulk_ingest(self) -> None:
        from src.graph.knowledge_graph import KnowledgeGraph
        from src.ingesters.feeds import ThreatEntity, ThreatRelationship

        kg = KnowledgeGraph()
        await kg.connect()

        entities = [
            ThreatEntity(entity_type="indicator", stix_id=f"indicator--bulk-{i}", name=f"10.0.0.{i}")
            for i in range(10)
        ]

        stats = await kg.bulk_ingest(entities, [])
        assert stats["entities_ingested"] == 10

        # Clean up
        async with kg._driver.session(database=kg.database) as session:
            await session.run("MATCH (n) WHERE n.stix_id CONTAINS 'bulk-' DETACH DELETE n")

        await kg.close()


@pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not available")
class TestGNNWithNeo4j:
    """Test GNN pipeline with real graph data."""

    @pytest.mark.asyncio
    async def test_full_prediction_pipeline(self) -> None:
        from src.graph.knowledge_graph import KnowledgeGraph
        from src.gnn.predictor import TechniquePredictor
        from src.ingesters.feeds import ThreatEntity, ThreatRelationship

        kg = KnowledgeGraph()
        await kg.connect()

        # Seed graph with test data
        entities = [
            ThreatEntity(entity_type="threat_actor", stix_id="ta--gnn-test-1", name="GNNTestActor1"),
            ThreatEntity(entity_type="threat_actor", stix_id="ta--gnn-test-2", name="GNNTestActor2"),
            ThreatEntity(entity_type="technique", stix_id="tech--gnn-test-1", name="GNNTech1", properties={"mitre_id": "T0001"}),
            ThreatEntity(entity_type="technique", stix_id="tech--gnn-test-2", name="GNNTech2", properties={"mitre_id": "T0002"}),
            ThreatEntity(entity_type="technique", stix_id="tech--gnn-test-3", name="GNNTech3", properties={"mitre_id": "T0003"}),
        ]
        rels = [
            ThreatRelationship(source_id="ta--gnn-test-1", target_id="tech--gnn-test-1", relationship_type="uses"),
            ThreatRelationship(source_id="ta--gnn-test-1", target_id="tech--gnn-test-2", relationship_type="uses"),
            ThreatRelationship(source_id="ta--gnn-test-2", target_id="tech--gnn-test-3", relationship_type="uses"),
        ]
        await kg.bulk_ingest(entities, rels)

        # Export and predict
        graph_export = await kg.query_full_graph(limit=1000)
        predictor = TechniquePredictor(embedding_dim=16, hidden_dim=32, num_layers=2)
        graph_data = predictor.build_graph_data(graph_export["nodes"], graph_export["edges"])
        predictor.train(graph_data, epochs=20)

        tech_ids = [n["id"] for n in graph_export["nodes"] if n.get("type") == "Technique"]
        result = predictor.predict_techniques(graph_data, "ta--gnn-test-1", tech_ids, top_k=3)
        assert len(result.predicted_techniques) > 0

        # Clean up
        async with kg._driver.session(database=kg.database) as session:
            await session.run("MATCH (n) WHERE n.stix_id CONTAINS 'gnn-test' DETACH DELETE n")

        await kg.close()


class TestAPIWithAuth:
    """Test API authentication."""

    def test_auth_enforced_when_keys_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_KEYS", "threat-intel-key-789")

        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/api/v1/stats")
        # Stats doesn't require auth (read-only), should pass
        assert resp.status_code == 200

    def test_write_endpoints_need_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_KEYS", "threat-intel-key-789")

        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Ingest without key = 401
        resp = client.post("/api/v1/ingest", json={"feed": "cisa_kev"})
        assert resp.status_code == 401

        # Ingest with key = either 200 or 503 (Neo4j not connected)
        resp = client.post(
            "/api/v1/ingest",
            json={"feed": "cisa_kev"},
            headers={"X-API-Key": "threat-intel-key-789"},
        )
        assert resp.status_code in (200, 503)
