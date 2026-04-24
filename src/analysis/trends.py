"""Temporal trend analysis for emerging attack pattern detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class TechniqueObservation:
    """A single observation of a technique being used."""

    technique_id: str
    technique_name: str
    timestamp: str
    actor_id: str | None = None
    campaign_id: str | None = None
    source_feed: str = ""


@dataclass
class TrendResult:
    """Result of trend analysis for a single technique."""

    technique_id: str
    technique_name: str
    observation_count: int
    trend_direction: str  # increasing, decreasing, stable
    trend_slope: float
    trend_p_value: float
    is_emerging: bool
    z_score: float
    daily_counts: dict[str, int]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendReport:
    """Complete trend analysis report."""

    window_start: str
    window_end: str
    total_observations: int
    emerging_techniques: list[TrendResult]
    declining_techniques: list[TrendResult]
    top_techniques: list[TrendResult]


class TemporalAnalyzer:
    """Analyzes temporal patterns in threat intelligence observations."""

    def __init__(
        self,
        window_days: int = 30,
        min_observations: int = 5,
        emerging_threshold: float = 2.0,
    ) -> None:
        self.window_days = window_days
        self.min_observations = min_observations
        self.emerging_threshold = emerging_threshold
        self._observations: list[TechniqueObservation] = []

    def add_observation(self, obs: TechniqueObservation) -> None:
        self._observations.append(obs)

    def add_observations(self, obs_list: list[TechniqueObservation]) -> None:
        self._observations.extend(obs_list)

    def analyze_technique(
        self,
        technique_id: str,
        reference_end: datetime | None = None,
    ) -> TrendResult | None:
        """Analyze temporal trend for a single technique."""
        end = reference_end or datetime.utcnow()
        start = end - timedelta(days=self.window_days)

        relevant = [
            o for o in self._observations
            if o.technique_id == technique_id
            and start <= datetime.fromisoformat(o.timestamp.replace("Z", "")) <= end
        ]

        if len(relevant) < self.min_observations:
            return None

        tech_name = relevant[0].technique_name

        # Build daily counts
        daily: dict[str, int] = {}
        for o in relevant:
            day = o.timestamp[:10]
            daily[day] = daily.get(day, 0) + 1

        # Fill missing days with 0
        current = start
        while current <= end:
            key = current.strftime("%Y-%m-%d")
            if key not in daily:
                daily[key] = 0
            current += timedelta(days=1)

        sorted_days = sorted(daily.keys())
        counts = np.array([daily[d] for d in sorted_days], dtype=np.float64)

        # Linear regression for trend
        x = np.arange(len(counts), dtype=np.float64)
        if len(x) < 2:
            return None

        slope, _, _, p_value, _ = stats.linregress(x, counts)

        # Z-score: compare recent activity to historical mean
        if len(counts) >= 7:
            recent = counts[-7:]
            historical = counts[:-7]
            if len(historical) > 0 and np.std(historical) > 0:
                z_score = float(
                    (np.mean(recent) - np.mean(historical)) / np.std(historical)
                )
            else:
                z_score = 0.0
        else:
            z_score = 0.0

        # Determine trend direction
        if p_value < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
        else:
            direction = "stable"

        is_emerging = z_score >= self.emerging_threshold and direction == "increasing"

        return TrendResult(
            technique_id=technique_id,
            technique_name=tech_name,
            observation_count=len(relevant),
            trend_direction=direction,
            trend_slope=round(float(slope), 4),
            trend_p_value=round(float(p_value), 4),
            is_emerging=is_emerging,
            z_score=round(z_score, 4),
            daily_counts=daily,
        )

    def generate_report(
        self,
        reference_end: datetime | None = None,
    ) -> TrendReport:
        """Generate a complete trend analysis report."""
        end = reference_end or datetime.utcnow()
        start = end - timedelta(days=self.window_days)

        # Get all techniques observed in window
        technique_ids = {
            o.technique_id
            for o in self._observations
            if start <= datetime.fromisoformat(o.timestamp.replace("Z", "")) <= end
        }

        all_trends: list[TrendResult] = []
        for tid in technique_ids:
            result = self.analyze_technique(tid, reference_end=end)
            if result:
                all_trends.append(result)

        emerging = [t for t in all_trends if t.is_emerging]
        emerging.sort(key=lambda t: t.z_score, reverse=True)

        declining = [
            t for t in all_trends
            if t.trend_direction == "decreasing" and t.trend_p_value < 0.05
        ]
        declining.sort(key=lambda t: t.trend_slope)

        top_by_count = sorted(all_trends, key=lambda t: t.observation_count, reverse=True)[:10]

        return TrendReport(
            window_start=start.isoformat(),
            window_end=end.isoformat(),
            total_observations=len(self._observations),
            emerging_techniques=emerging,
            declining_techniques=declining,
            top_techniques=top_by_count,
        )

    def get_co_occurrence_matrix(
        self,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Compute technique co-occurrence within campaigns/actors."""
        # Group techniques by actor
        actor_techniques: dict[str, set[str]] = {}
        for o in self._observations:
            if o.actor_id:
                actor_techniques.setdefault(o.actor_id, set()).add(o.technique_id)

        # Count co-occurrences
        technique_counts = Counter(o.technique_id for o in self._observations)
        top_techniques = [t for t, _ in technique_counts.most_common(top_n)]

        co_matrix: dict[str, dict[str, int]] = {t: {} for t in top_techniques}
        for techniques in actor_techniques.values():
            relevant = [t for t in techniques if t in top_techniques]
            for i, t1 in enumerate(relevant):
                for t2 in relevant[i + 1 :]:
                    co_matrix[t1][t2] = co_matrix[t1].get(t2, 0) + 1
                    co_matrix[t2][t1] = co_matrix[t2].get(t1, 0) + 1

        return {
            "techniques": top_techniques,
            "matrix": co_matrix,
        }
