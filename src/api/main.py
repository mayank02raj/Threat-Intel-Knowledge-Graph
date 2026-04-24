"""FastAPI query interface for the Threat Intelligence Knowledge Graph."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.middleware import RateLimiter, require_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_RPM", "60")),
    window_seconds=60,
)


# --- Application state managed via lifespan ---
class AppState:
    """Shared application state for Neo4j connection and GNN model cache."""

    def __init__(self) -> None:
        self.kg: Any = None
        self.predictor: Any = None
        self._model_trained: bool = False


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage Neo4j connection lifecycle: connect on startup, close on shutdown."""
    from src.graph.knowledge_graph import KnowledgeGraph

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        _state.kg = KnowledgeGraph(
            uri=neo4j_uri, user=neo4j_user, password=neo4j_password
        )
        await _state.kg.connect()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.warning(f"Neo4j connection failed (non-fatal): {e}")
        _state.kg = None

    yield

    if _state.kg:
        await _state.kg.close()
        logger.info("Neo4j connection closed")


app = FastAPI(
    title="Threat Intelligence Knowledge Graph",
    description="Real-time threat intelligence knowledge graph with GNN prediction",
    version="0.1.0",
    lifespan=lifespan,
)

_allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---

class IngestRequest(BaseModel):
    feed: str = Field(..., description="Feed name: otx, misp, or cisa_kev")
    api_key: str | None = None
    url: str | None = None


class IngestResponse(BaseModel):
    feed: str
    success: bool
    entities_ingested: int
    relationships_ingested: int
    error: str | None = None


class GraphQueryRequest(BaseModel):
    query_type: str = Field(..., description="actor_techniques, technique_actors, or full_graph")
    actor_name: str | None = None
    technique_id: str | None = None
    limit: int = Field(default=100, le=1000)


class PredictionRequest(BaseModel):
    actor_id: str
    top_k: int = Field(default=10, le=50)


class PredictionResponse(BaseModel):
    actor_id: str
    actor_name: str
    predicted_techniques: list[dict[str, Any]]


class TrendRequest(BaseModel):
    window_days: int = Field(default=30, le=365)
    min_observations: int = Field(default=5, le=100)


class HealthResponse(BaseModel):
    status: str
    version: str
    neo4j_connected: bool
    graph_stats: dict[str, int] | None = None


# --- Endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    _rate_limiter.check(request)
    connected = _state.kg is not None
    stats = None
    if connected:
        try:
            stats = await _state.kg.get_stats()
        except Exception:
            connected = False

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        neo4j_connected=connected,
        graph_stats=stats,
    )


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_feed(
    request_data: IngestRequest,
    request: Request,
    _key: str = Depends(require_api_key),
) -> IngestResponse:
    """Trigger ingestion from a threat intelligence feed."""
    _rate_limiter.check(request)
    if not _state.kg:
        raise HTTPException(503, "Neo4j not connected")

    try:
        from src.ingesters.feeds import CISAKEVIngester, OTXIngester

        if request_data.feed == "cisa_kev":
            ingester = CISAKEVIngester()
        elif request_data.feed == "otx":
            if not request_data.api_key:
                raise HTTPException(400, "OTX requires an API key")
            ingester = OTXIngester(api_key=request_data.api_key)
        else:
            raise HTTPException(400, f"Unknown feed: {request_data.feed}")

        result = await ingester.ingest()

        if not result.success:
            return IngestResponse(
                feed=request_data.feed,
                success=False,
                entities_ingested=0,
                relationships_ingested=0,
                error=result.error,
            )

        # Store in Neo4j using shared connection
        stats = await _state.kg.bulk_ingest(result.entities, result.relationships)

        # Invalidate cached GNN model since graph changed
        _state._model_trained = False

        return IngestResponse(
            feed=request_data.feed,
            success=True,
            entities_ingested=stats["entities_ingested"],
            relationships_ingested=stats["relationships_ingested"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return IngestResponse(
            feed=request_data.feed,
            success=False,
            entities_ingested=0,
            relationships_ingested=0,
            error=str(e),
        )


@app.post("/api/v1/graph/query")
async def query_graph(
    query: GraphQueryRequest,
    request: Request,
    _key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Query the knowledge graph."""
    _rate_limiter.check(request)
    if not _state.kg:
        raise HTTPException(503, "Neo4j not connected")

    try:
        if query.query_type == "actor_techniques" and query.actor_name:
            results = await _state.kg.query_actor_techniques(query.actor_name)
            return {"query_type": query.query_type, "results": results}

        elif query.query_type == "technique_actors" and query.technique_id:
            results = await _state.kg.query_technique_actors(query.technique_id)
            return {"query_type": query.query_type, "results": results}

        elif query.query_type == "full_graph":
            results = await _state.kg.query_full_graph(limit=query.limit)
            return {"query_type": query.query_type, **results}

        else:
            raise HTTPException(400, "Invalid query_type or missing parameters")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_techniques(
    pred_request: PredictionRequest,
    request: Request,
    _key: str = Depends(require_api_key),
) -> PredictionResponse:
    """Predict next likely techniques for a threat actor using the GNN.

    The model is trained once and cached. It retrains only when the graph
    changes (after ingestion) or on explicit request.
    """
    _rate_limiter.check(request)
    if not _state.kg:
        raise HTTPException(503, "Neo4j not connected")

    try:
        from src.gnn.predictor import TechniquePredictor

        graph_export = await _state.kg.query_full_graph(limit=5000)

        if not graph_export.get("nodes"):
            raise HTTPException(400, "Graph is empty. Ingest data first.")

        # Train only if model is not cached or graph has changed
        if _state.predictor is None or not _state._model_trained:
            logger.info("Training GNN model (first run or graph updated)...")
            _state.predictor = TechniquePredictor(
                embedding_dim=64, hidden_dim=128, num_layers=2
            )
            graph_data = _state.predictor.build_graph_data(
                graph_export["nodes"], graph_export["edges"]
            )
            _state.predictor.train(graph_data, epochs=100)
            _state._model_trained = True
            _state._cached_graph_data = graph_data
            _state._cached_technique_ids = [
                n["id"] for n in graph_export["nodes"] if n.get("type") == "Technique"
            ]
            logger.info("GNN model trained and cached")

        result = _state.predictor.predict_techniques(
            _state._cached_graph_data,
            pred_request.actor_id,
            _state._cached_technique_ids,
            top_k=pred_request.top_k,
        )

        return PredictionResponse(
            actor_id=result.actor_id,
            actor_name=result.actor_name,
            predicted_techniques=result.predicted_techniques,
        )

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/predict/retrain")
async def retrain_model(
    request: Request,
    _key: str = Depends(require_api_key),
) -> dict[str, str]:
    """Force GNN model retraining."""
    _rate_limiter.check(request)
    _state._model_trained = False
    return {"status": "Model cache invalidated. Next prediction will retrain."}


@app.get("/api/v1/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    """Get graph statistics."""
    _rate_limiter.check(request)
    if not _state.kg:
        return {"error": "Neo4j not connected", "hint": "Check NEO4J_URI env var"}
    try:
        return await _state.kg.get_stats()
    except Exception as e:
        return {"error": str(e)}
