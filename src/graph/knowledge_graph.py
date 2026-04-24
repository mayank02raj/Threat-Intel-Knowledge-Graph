"""Neo4j knowledge graph manager for threat intelligence entities."""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from src.exceptions import GraphError
from src.ingesters.feeds import ThreatEntity, ThreatRelationship

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Manages the Neo4j threat intelligence knowledge graph."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            await self._create_constraints()
        except Exception as e:
            raise GraphError("connect", f"Failed to connect to Neo4j at {self.uri}: {e}") from e

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _create_constraints(self) -> None:
        """Create uniqueness constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ThreatActor) REQUIRE n.stix_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Technique) REQUIRE n.stix_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Malware) REQUIRE n.stix_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Indicator) REQUIRE n.stix_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Campaign) REQUIRE n.stix_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vulnerability) REQUIRE n.stix_id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (n:Technique) ON (n.mitre_id)",
        ]
        assert self._driver is not None
        async with self._driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")

    def _entity_type_to_label(self, entity_type: str) -> str:
        """Map entity types to Neo4j labels."""
        mapping = {
            "threat_actor": "ThreatActor",
            "technique": "Technique",
            "malware": "Malware",
            "indicator": "Indicator",
            "campaign": "Campaign",
            "vulnerability": "Vulnerability",
        }
        return mapping.get(entity_type, "Entity")

    def _rel_type_to_neo4j(self, rel_type: str) -> str:
        """Map relationship types to Neo4j relationship types."""
        mapping = {
            "uses": "USES",
            "deploys": "DEPLOYS",
            "indicates": "INDICATES",
            "attributed_to": "ATTRIBUTED_TO",
            "subtechnique_of": "SUBTECHNIQUE_OF",
            "targets": "TARGETS",
            "mitigates": "MITIGATES",
        }
        return mapping.get(rel_type, "RELATED_TO")

    async def upsert_entity(self, entity: ThreatEntity) -> None:
        """Insert or update an entity in the graph."""
        assert self._driver is not None
        label = self._entity_type_to_label(entity.entity_type)
        props = {
            "stix_id": entity.stix_id,
            "name": entity.name,
            "description": entity.description,
            "labels": entity.labels,
            "created": entity.created,
            "modified": entity.modified,
            **entity.properties,
        }

        query = f"""
        MERGE (n:{label} {{stix_id: $stix_id}})
        SET n += $props
        """

        async with self._driver.session(database=self.database) as session:
            await session.run(query, stix_id=entity.stix_id, props=props)

    async def upsert_relationship(self, rel: ThreatRelationship) -> None:
        """Insert or update a relationship in the graph."""
        assert self._driver is not None
        rel_type = self._rel_type_to_neo4j(rel.relationship_type)

        query = f"""
        MATCH (a {{stix_id: $source_id}})
        MATCH (b {{stix_id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.description = $description,
            r.first_seen = $first_seen,
            r.last_seen = $last_seen
        """

        async with self._driver.session(database=self.database) as session:
            await session.run(
                query,
                source_id=rel.source_id,
                target_id=rel.target_id,
                description=rel.description,
                first_seen=rel.first_seen,
                last_seen=rel.last_seen,
            )

    async def bulk_ingest(
        self,
        entities: list[ThreatEntity],
        relationships: list[ThreatRelationship],
    ) -> dict[str, int]:
        """Bulk ingest entities and relationships."""
        entity_count = 0
        rel_count = 0

        for entity in entities:
            try:
                await self.upsert_entity(entity)
                entity_count += 1
            except Exception as e:
                logger.error(f"Entity upsert error for {entity.stix_id}: {e}")

        for rel in relationships:
            try:
                await self.upsert_relationship(rel)
                rel_count += 1
            except Exception as e:
                logger.error(f"Relationship upsert error: {e}")

        return {"entities_ingested": entity_count, "relationships_ingested": rel_count}

    async def query_actor_techniques(self, actor_name: str) -> list[dict[str, Any]]:
        """Get all techniques used by a threat actor."""
        assert self._driver is not None
        query = """
        MATCH (a:ThreatActor {name: $name})-[:USES]->(t:Technique)
        RETURN t.stix_id AS stix_id, t.name AS name, t.mitre_id AS mitre_id
        ORDER BY t.name
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, name=actor_name)
            return [dict(record) async for record in result]

    async def query_technique_actors(self, technique_id: str) -> list[dict[str, Any]]:
        """Get all actors that use a specific technique."""
        assert self._driver is not None
        query = """
        MATCH (a:ThreatActor)-[:USES]->(t:Technique {mitre_id: $technique_id})
        RETURN a.stix_id AS stix_id, a.name AS name
        ORDER BY a.name
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, technique_id=technique_id)
            return [dict(record) async for record in result]

    async def query_full_graph(self, limit: int = 500) -> dict[str, Any]:
        """Export graph structure for visualization."""
        assert self._driver is not None
        node_query = """
        MATCH (n)
        RETURN n.stix_id AS id, n.name AS name, labels(n)[0] AS type
        LIMIT $limit
        """
        edge_query = """
        MATCH (a)-[r]->(b)
        RETURN a.stix_id AS source, b.stix_id AS target, type(r) AS rel_type
        LIMIT $limit
        """
        async with self._driver.session(database=self.database) as session:
            node_result = await session.run(node_query, limit=limit)
            nodes = [dict(r) async for r in node_result]

            edge_result = await session.run(edge_query, limit=limit)
            edges = [dict(r) async for r in edge_result]

        return {"nodes": nodes, "edges": edges}

    async def get_stats(self) -> dict[str, int]:
        """Get graph statistics."""
        assert self._driver is not None
        queries = {
            "threat_actors": "MATCH (n:ThreatActor) RETURN count(n) AS c",
            "techniques": "MATCH (n:Technique) RETURN count(n) AS c",
            "malware": "MATCH (n:Malware) RETURN count(n) AS c",
            "indicators": "MATCH (n:Indicator) RETURN count(n) AS c",
            "campaigns": "MATCH (n:Campaign) RETURN count(n) AS c",
            "relationships": "MATCH ()-[r]->() RETURN count(r) AS c",
        }
        stats: dict[str, int] = {}
        async with self._driver.session(database=self.database) as session:
            for key, query in queries.items():
                result = await session.run(query)
                record = await result.single()
                stats[key] = record["c"] if record else 0
        return stats
