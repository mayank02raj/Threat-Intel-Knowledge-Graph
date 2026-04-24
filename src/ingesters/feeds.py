"""STIX/TAXII feed ingesters for AlienVault OTX, MISP, and CISA KEV."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from src.exceptions import IngestionError

logger = logging.getLogger(__name__)


@dataclass
class ThreatEntity:
    """Normalized threat intelligence entity."""

    entity_type: str  # threat_actor, technique, malware, indicator, campaign
    stix_id: str
    name: str
    description: str = ""
    labels: list[str] = field(default_factory=list)
    external_references: list[dict[str, str]] = field(default_factory=list)
    created: str = ""
    modified: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatRelationship:
    """Normalized relationship between threat entities."""

    source_id: str
    target_id: str
    relationship_type: str  # uses, deploys, indicates, attributed_to
    description: str = ""
    first_seen: str = ""
    last_seen: str = ""


@dataclass
class IngestResult:
    """Result of a feed ingestion run."""

    feed_name: str
    entities: list[ThreatEntity]
    relationships: list[ThreatRelationship]
    timestamp: str
    success: bool
    error: str | None = None
    stats: dict[str, int] = field(default_factory=dict)


class FeedIngester(ABC):
    """Abstract base class for threat intelligence feed ingesters."""

    feed_name: str = "base"

    @abstractmethod
    async def ingest(self) -> IngestResult:
        """Fetch and normalize data from the feed."""
        ...


class OTXIngester(FeedIngester):
    """AlienVault OTX feed ingester."""

    feed_name = "otx"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://otx.alienvault.com/api/v1",
        max_pulses: int = 100,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.max_pulses = max_pulses

    async def ingest(self) -> IngestResult:
        entities: list[ThreatEntity] = []
        relationships: list[ThreatRelationship] = []

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(
                    f"{self.base_url}/pulses/subscribed",
                    headers={"X-OTX-API-KEY": self.api_key},
                    params={"limit": self.max_pulses, "modified_since": ""},
                )
                resp.raise_for_status()
                data = resp.json()

            for pulse in data.get("results", [])[:self.max_pulses]:
                pulse_id = f"otx-pulse-{pulse['id']}"

                # Create campaign entity from pulse
                campaign = ThreatEntity(
                    entity_type="campaign",
                    stix_id=pulse_id,
                    name=pulse.get("name", "Unknown"),
                    description=pulse.get("description", ""),
                    labels=pulse.get("tags", []),
                    created=pulse.get("created", ""),
                    modified=pulse.get("modified", ""),
                    properties={"adversary": pulse.get("adversary", "")},
                )
                entities.append(campaign)

                # Extract indicators
                for indicator in pulse.get("indicators", []):
                    ioc_id = f"otx-ioc-{indicator['id']}"
                    ioc = ThreatEntity(
                        entity_type="indicator",
                        stix_id=ioc_id,
                        name=indicator.get("indicator", ""),
                        description=indicator.get("description", ""),
                        labels=[indicator.get("type", "unknown")],
                        created=indicator.get("created", ""),
                        properties={"ioc_type": indicator.get("type", "")},
                    )
                    entities.append(ioc)
                    relationships.append(
                        ThreatRelationship(
                            source_id=pulse_id,
                            target_id=ioc_id,
                            relationship_type="indicates",
                        )
                    )

                # Extract ATT&CK techniques
                for attack_id in pulse.get("attack_ids", []):
                    tech_id = f"attack-pattern--{attack_id['id']}"
                    tech = ThreatEntity(
                        entity_type="technique",
                        stix_id=tech_id,
                        name=attack_id.get("display_name", attack_id["id"]),
                        properties={"mitre_id": attack_id["id"]},
                    )
                    entities.append(tech)
                    relationships.append(
                        ThreatRelationship(
                            source_id=pulse_id,
                            target_id=tech_id,
                            relationship_type="uses",
                        )
                    )

            return IngestResult(
                feed_name=self.feed_name,
                entities=entities,
                relationships=relationships,
                timestamp=datetime.utcnow().isoformat(),
                success=True,
                stats={"pulses": len(data.get("results", [])), "entities": len(entities)},
            )

        except Exception as e:
            logger.error(f"OTX ingestion error: {e}")
            return IngestResult(
                feed_name=self.feed_name,
                entities=[],
                relationships=[],
                timestamp=datetime.utcnow().isoformat(),
                success=False,
                error=str(e),
            )


class CISAKEVIngester(FeedIngester):
    """CISA Known Exploited Vulnerabilities catalog ingester."""

    feed_name = "cisa_kev"

    def __init__(
        self,
        catalog_url: str = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    ) -> None:
        self.catalog_url = catalog_url

    async def ingest(self) -> IngestResult:
        entities: list[ThreatEntity] = []
        relationships: list[ThreatRelationship] = []

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(self.catalog_url)
                resp.raise_for_status()
                data = resp.json()

            for vuln in data.get("vulnerabilities", []):
                cve_id = vuln.get("cveID", "")
                entity = ThreatEntity(
                    entity_type="vulnerability",
                    stix_id=f"vulnerability--{cve_id}",
                    name=cve_id,
                    description=vuln.get("shortDescription", ""),
                    labels=["known-exploited"],
                    properties={
                        "vendor": vuln.get("vendorProject", ""),
                        "product": vuln.get("product", ""),
                        "date_added": vuln.get("dateAdded", ""),
                        "due_date": vuln.get("dueDate", ""),
                        "required_action": vuln.get("requiredAction", ""),
                        "known_ransomware": vuln.get("knownRansomwareCampaignUse", "Unknown"),
                    },
                )
                entities.append(entity)

            return IngestResult(
                feed_name=self.feed_name,
                entities=entities,
                relationships=relationships,
                timestamp=datetime.utcnow().isoformat(),
                success=True,
                stats={"vulnerabilities": len(entities)},
            )

        except Exception as e:
            logger.error(f"CISA KEV ingestion error: {e}")
            return IngestResult(
                feed_name=self.feed_name,
                entities=[],
                relationships=[],
                timestamp=datetime.utcnow().isoformat(),
                success=False,
                error=str(e),
            )


class MISPIngester(FeedIngester):
    """MISP threat sharing platform ingester."""

    feed_name = "misp"

    def __init__(
        self,
        url: str,
        api_key: str,
        verify_ssl: bool = False,
    ) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.verify_ssl = verify_ssl

    async def ingest(self) -> IngestResult:
        entities: list[ThreatEntity] = []
        relationships: list[ThreatRelationship] = []

        try:
            async with httpx.AsyncClient(timeout=60, verify=self.verify_ssl) as client:
                resp = await client.post(
                    f"{self.url}/events/restSearch",
                    headers={
                        "Authorization": self.api_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json={"limit": 100, "published": True},
                )
                resp.raise_for_status()
                data = resp.json()

            for event_wrapper in data.get("response", []):
                event = event_wrapper.get("Event", {})
                event_id = f"misp-event-{event.get('id', '')}"

                campaign = ThreatEntity(
                    entity_type="campaign",
                    stix_id=event_id,
                    name=event.get("info", "Unknown"),
                    description=event.get("info", ""),
                    labels=event.get("Tag", []),
                    created=event.get("date", ""),
                    properties={
                        "threat_level": event.get("threat_level_id", ""),
                        "analysis": event.get("analysis", ""),
                    },
                )
                entities.append(campaign)

                for attr in event.get("Attribute", []):
                    attr_id = f"misp-attr-{attr.get('id', '')}"
                    ioc = ThreatEntity(
                        entity_type="indicator",
                        stix_id=attr_id,
                        name=attr.get("value", ""),
                        labels=[attr.get("category", ""), attr.get("type", "")],
                        properties={
                            "category": attr.get("category", ""),
                            "ioc_type": attr.get("type", ""),
                            "to_ids": attr.get("to_ids", False),
                        },
                    )
                    entities.append(ioc)
                    relationships.append(
                        ThreatRelationship(
                            source_id=event_id,
                            target_id=attr_id,
                            relationship_type="indicates",
                        )
                    )

            return IngestResult(
                feed_name=self.feed_name,
                entities=entities,
                relationships=relationships,
                timestamp=datetime.utcnow().isoformat(),
                success=True,
                stats={"events": len(data.get("response", [])), "entities": len(entities)},
            )

        except Exception as e:
            logger.error(f"MISP ingestion error: {e}")
            return IngestResult(
                feed_name=self.feed_name,
                entities=[],
                relationships=[],
                timestamp=datetime.utcnow().isoformat(),
                success=False,
                error=str(e),
            )
