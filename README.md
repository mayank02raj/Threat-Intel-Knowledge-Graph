# Threat Intelligence Knowledge Graph

[![CI](https://github.com/mayank02raj/threat-intel-knowledge-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/threat-intel-knowledge-graph/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time threat intelligence knowledge graph platform that ingests STIX/TAXII feeds, builds a Neo4j knowledge graph mapping threat actors to techniques to malware to IoCs, and uses Graph Neural Networks to predict likely next techniques per threat actor.

## Key Features

- **STIX/TAXII Feed Ingestion**: AlienVault OTX, MISP, CISA Known Exploited Vulnerabilities
- **Neo4j Knowledge Graph**: Full entity relationship mapping (actors, techniques, malware, IoCs)
- **GNN Prediction Engine**: PyTorch Geometric model predicting next likely ATT&CK techniques per threat actor
- **Temporal Trend Analysis**: Emerging attack pattern detection with time-series analysis
- **Interactive Visualization**: Pyvis-powered graph exploration in Streamlit dashboard
- **Production-Ready**: FastAPI query interface, Docker Compose, CI/CD

## Architecture

```
+----------------+     +------------------+     +----------------+
| STIX/TAXII     |---->|  Feed Ingesters  |---->|  Neo4j Graph   |
| Feeds (OTX,    |     |  (normalize to   |     |  Database      |
|  MISP, CISA)   |     |   STIX 2.1)      |     +-------+--------+
+----------------+     +------------------+             |
                                                        v
+----------------+     +------------------+     +----------------+
| Streamlit      |<----|  FastAPI Query    |<----|  GNN Predictor |
| Dashboard      |     |  Interface       |     |  (PyG model)   |
+----------------+     +------------------+     +----------------+
```

## Quickstart

```bash
docker compose up --build
# Neo4j:     http://localhost:7474 (neo4j/password)
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start Neo4j (requires local instance or Docker)
docker run -d -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5

uvicorn src.api.main:app --reload --port 8000
streamlit run src/dashboard/app.py
pytest tests/ -v --cov=src
```

## Graph Schema

- **ThreatActor** -[USES]-> **Technique** (MITRE ATT&CK)
- **ThreatActor** -[DEPLOYS]-> **Malware**
- **Malware** -[INDICATES]-> **Indicator** (IoC)
- **Technique** -[SUBTECHNIQUE_OF]-> **Technique**
- **Campaign** -[ATTRIBUTED_TO]-> **ThreatActor**
- **Campaign** -[USES]-> **Technique**

## GNN Technique Prediction

The graph neural network model learns embeddings from the knowledge graph structure and predicts which ATT&CK techniques a threat actor is likely to adopt next, based on historical patterns and graph neighborhood signals.

## License

MIT
