"""Tests for the Threat Intelligence Knowledge Graph."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.analysis.trends import TemporalAnalyzer, TechniqueObservation, TrendResult
from src.gnn.predictor import ThreatGNN, TechniquePredictor
from src.ingesters.feeds import (
    CISAKEVIngester,
    ThreatEntity,
    ThreatRelationship,
    IngestResult,
)


# --- Feed Ingester Tests ---

class TestThreatEntities:
    def test_entity_creation(self) -> None:
        entity = ThreatEntity(
            entity_type="threat_actor",
            stix_id="threat-actor--abc123",
            name="APT28",
            description="Russian state-sponsored group",
            labels=["nation-state"],
        )
        assert entity.entity_type == "threat_actor"
        assert entity.name == "APT28"
        assert entity.stix_id == "threat-actor--abc123"

    def test_relationship_creation(self) -> None:
        rel = ThreatRelationship(
            source_id="threat-actor--abc",
            target_id="attack-pattern--xyz",
            relationship_type="uses",
        )
        assert rel.relationship_type == "uses"

    def test_ingest_result(self) -> None:
        result = IngestResult(
            feed_name="test",
            entities=[
                ThreatEntity(entity_type="indicator", stix_id="ioc-1", name="1.2.3.4")
            ],
            relationships=[],
            timestamp="2024-01-01T00:00:00",
            success=True,
            stats={"indicators": 1},
        )
        assert result.success
        assert len(result.entities) == 1

    def test_cisa_ingester_init(self) -> None:
        ingester = CISAKEVIngester()
        assert ingester.feed_name == "cisa_kev"
        assert "cisa.gov" in ingester.catalog_url


# --- GNN Model Tests ---

class TestThreatGNN:
    def test_model_forward(self) -> None:
        model = ThreatGNN(num_node_features=16, embedding_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_label_index = torch.tensor([[0, 5], [3, 7]], dtype=torch.long)

        output = model(x, edge_index, edge_label_index)
        assert output.shape == (2,)

    def test_model_encode(self) -> None:
        model = ThreatGNN(num_node_features=16, embedding_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        embeddings = model.encode(x, edge_index)
        assert embeddings.shape == (10, 32)

    def test_model_decode(self) -> None:
        model = ThreatGNN(num_node_features=16, embedding_dim=32, hidden_dim=64, num_layers=2)
        z = torch.randn(10, 32)
        edge_label_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        scores = model.decode(z, edge_label_index)
        assert scores.shape == (2,)


class TestTechniquePredictor:
    @pytest.fixture
    def sample_graph(self) -> tuple[list[dict], list[dict]]:
        nodes = [
            {"id": "actor-1", "name": "APT28", "type": "ThreatActor"},
            {"id": "actor-2", "name": "APT29", "type": "ThreatActor"},
            {"id": "tech-1", "name": "Phishing", "type": "Technique"},
            {"id": "tech-2", "name": "Command Line", "type": "Technique"},
            {"id": "tech-3", "name": "Valid Accounts", "type": "Technique"},
            {"id": "mal-1", "name": "Emotet", "type": "Malware"},
        ]
        edges = [
            {"source": "actor-1", "target": "tech-1", "rel_type": "USES"},
            {"source": "actor-1", "target": "tech-2", "rel_type": "USES"},
            {"source": "actor-2", "target": "tech-3", "rel_type": "USES"},
            {"source": "actor-1", "target": "mal-1", "rel_type": "DEPLOYS"},
        ]
        return nodes, edges

    def test_build_graph_data(self, sample_graph: tuple[list, list]) -> None:
        nodes, edges = sample_graph
        predictor = TechniquePredictor(embedding_dim=32, hidden_dim=64, num_layers=2)
        data = predictor.build_graph_data(nodes, edges)

        assert data.x.shape[0] == 6  # 6 nodes
        assert data.edge_index.shape[1] == 4  # 4 edges

    def test_train(self, sample_graph: tuple[list, list]) -> None:
        nodes, edges = sample_graph
        predictor = TechniquePredictor(embedding_dim=16, hidden_dim=32, num_layers=2)
        data = predictor.build_graph_data(nodes, edges)

        losses = predictor.train(data, epochs=10)
        assert len(losses) == 10
        assert all(isinstance(l, float) for l in losses)

    def test_predict(self, sample_graph: tuple[list, list]) -> None:
        nodes, edges = sample_graph
        predictor = TechniquePredictor(embedding_dim=16, hidden_dim=32, num_layers=2)
        data = predictor.build_graph_data(nodes, edges)
        predictor.train(data, epochs=10)

        technique_ids = ["tech-1", "tech-2", "tech-3"]
        result = predictor.predict_techniques(data, "actor-1", technique_ids, top_k=3)

        assert result.actor_id == "actor-1"
        assert len(result.predicted_techniques) <= 3
        for pred in result.predicted_techniques:
            assert 0.0 <= pred["probability"] <= 1.0

    def test_predict_unknown_actor(self, sample_graph: tuple[list, list]) -> None:
        nodes, edges = sample_graph
        predictor = TechniquePredictor(embedding_dim=16, hidden_dim=32, num_layers=2)
        data = predictor.build_graph_data(nodes, edges)
        predictor.train(data, epochs=5)

        with pytest.raises(ValueError, match="not found"):
            predictor.predict_techniques(data, "nonexistent", ["tech-1"])


# --- Temporal Analysis Tests ---

class TestTemporalAnalyzer:
    def _make_observations(
        self, technique_id: str, name: str, n: int, base_date: datetime, trend: float = 0.0
    ) -> list[TechniqueObservation]:
        obs = []
        for i in range(n):
            count = max(1, int(5 + trend * i + np.random.normal(0, 1)))
            for _ in range(count):
                ts = (base_date + timedelta(days=i)).isoformat()
                obs.append(TechniqueObservation(
                    technique_id=technique_id,
                    technique_name=name,
                    timestamp=ts,
                    actor_id=f"actor-{np.random.randint(1, 5)}",
                ))
        return obs

    def test_analyzer_init(self) -> None:
        analyzer = TemporalAnalyzer(window_days=30)
        assert analyzer.window_days == 30

    def test_analyze_technique_emerging(self) -> None:
        analyzer = TemporalAnalyzer(window_days=30, min_observations=5, emerging_threshold=1.5)
        base = datetime.utcnow() - timedelta(days=30)
        obs = self._make_observations("T1566", "Phishing", 30, base, trend=0.5)
        analyzer.add_observations(obs)

        result = analyzer.analyze_technique("T1566")
        assert result is not None
        assert isinstance(result, TrendResult)
        assert result.observation_count > 0

    def test_analyze_insufficient_data(self) -> None:
        analyzer = TemporalAnalyzer(min_observations=100)
        analyzer.add_observation(TechniqueObservation(
            technique_id="T1234",
            technique_name="Test",
            timestamp=datetime.utcnow().isoformat(),
        ))
        result = analyzer.analyze_technique("T1234")
        assert result is None

    def test_generate_report(self) -> None:
        analyzer = TemporalAnalyzer(window_days=30, min_observations=3)
        base = datetime.utcnow() - timedelta(days=30)
        obs1 = self._make_observations("T1566", "Phishing", 30, base, trend=0.3)
        obs2 = self._make_observations("T1059", "Command Line", 30, base, trend=-0.2)
        analyzer.add_observations(obs1 + obs2)

        report = analyzer.generate_report()
        assert report.total_observations > 0
        assert len(report.top_techniques) > 0

    def test_co_occurrence(self) -> None:
        analyzer = TemporalAnalyzer()
        now = datetime.utcnow().isoformat()
        for actor in ["actor-1", "actor-2"]:
            for tech in ["T1", "T2", "T3"]:
                analyzer.add_observation(TechniqueObservation(
                    technique_id=tech, technique_name=tech,
                    timestamp=now, actor_id=actor,
                ))
        matrix = analyzer.get_co_occurrence_matrix()
        assert len(matrix["techniques"]) > 0


# --- API Tests ---

class TestAPI:
    def test_health(self) -> None:
        from src.api.main import app

        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_stats_endpoint(self) -> None:
        from src.api.main import app

        client = TestClient(app)
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200

    def test_ingest_unknown_feed(self) -> None:
        from src.api.main import app

        client = TestClient(app)
        resp = client.post("/api/v1/ingest", json={"feed": "unknown_feed"})
        assert resp.status_code == 400


# --- Exception Tests ---

class TestExceptions:
    def test_base_error(self) -> None:
        from src.exceptions import TIKGError
        e = TIKGError("msg", {"k": "v"})
        assert str(e) == "msg"
        assert e.details == {"k": "v"}

    def test_ingestion_error(self) -> None:
        from src.exceptions import IngestionError
        e = IngestionError("otx", "api key invalid")
        assert e.feed_name == "otx"

    def test_graph_error(self) -> None:
        from src.exceptions import GraphError
        e = GraphError("upsert", "connection refused")
        assert e.operation == "upsert"

    def test_prediction_error(self) -> None:
        from src.exceptions import PredictionError
        e = PredictionError("apt28", "model not trained")
        assert e.actor_id == "apt28"

    def test_analysis_error(self) -> None:
        from src.exceptions import AnalysisError
        e = AnalysisError("T1566", "insufficient data")
        assert e.technique_id == "T1566"
