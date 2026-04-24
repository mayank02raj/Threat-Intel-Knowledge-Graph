"""Graph Neural Network for predicting next likely ATT&CK techniques per threat actor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.exceptions import PredictionError
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


@dataclass
class PredictionResult:
    """Result of technique prediction for a threat actor."""

    actor_id: str
    actor_name: str
    predicted_techniques: list[dict[str, Any]]  # [{technique_id, name, probability}]
    embedding: list[float] = field(default_factory=list)


class ThreatGNN(nn.Module):
    """Graph Neural Network for threat intelligence link prediction.

    Architecture: GraphSAGE encoder with bilinear decoder for predicting
    actor-technique links.
    """

    def __init__(
        self,
        num_node_features: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Encoder layers (GraphSAGE)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_node_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, embedding_dim))

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

        # Bilinear decoder for link prediction
        self.decoder = nn.Bilinear(embedding_dim, embedding_dim, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features into embeddings."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def decode(
        self, z: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """Decode embeddings to predict link probabilities."""
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return self.decoder(src, dst).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class TechniquePredictor:
    """High-level interface for training and inference with the GNN model."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        self._model: ThreatGNN | None = None
        self._node_mapping: dict[str, int] = {}
        self._reverse_mapping: dict[int, str] = {}
        self._node_names: dict[str, str] = {}

    def build_graph_data(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> Data:
        """Convert graph export to PyTorch Geometric Data object."""
        # Build node mapping
        self._node_mapping = {}
        self._reverse_mapping = {}
        self._node_names = {}

        for i, node in enumerate(nodes):
            node_id = node["id"]
            self._node_mapping[node_id] = i
            self._reverse_mapping[i] = node_id
            self._node_names[node_id] = node.get("name", node_id)

        num_nodes = len(nodes)

        # Node features: one-hot entity type + random init
        type_set = sorted({n.get("type", "Unknown") for n in nodes})
        type_to_idx = {t: i for i, t in enumerate(type_set)}
        num_types = len(type_set)

        x = torch.zeros(num_nodes, num_types + 32)
        for i, node in enumerate(nodes):
            node_type = node.get("type", "Unknown")
            x[i, type_to_idx[node_type]] = 1.0
        # Add random features for expressiveness
        x[:, num_types:] = torch.randn(num_nodes, 32) * 0.1

        # Build edge index
        src_list: list[int] = []
        dst_list: list[int] = []
        for edge in edges:
            s = self._node_mapping.get(edge["source"])
            t = self._node_mapping.get(edge["target"])
            if s is not None and t is not None:
                src_list.append(s)
                dst_list.append(t)

        if not src_list:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def train(
        self,
        graph_data: Data,
        epochs: int = 200,
        neg_sampling_ratio: float = 1.0,
    ) -> list[float]:
        """Train the GNN on the knowledge graph with link prediction objective."""
        num_features = graph_data.x.shape[1]
        self._model = ThreatGNN(
            num_node_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )

        graph_data = graph_data.to(self.device)
        losses: list[float] = []

        for epoch in range(epochs):
            self._model.train()
            optimizer.zero_grad()

            # Positive edges
            pos_edge_index = graph_data.edge_index
            num_pos = pos_edge_index.shape[1]

            if num_pos == 0:
                losses.append(0.0)
                continue

            # Negative sampling
            num_neg = int(num_pos * neg_sampling_ratio)
            num_nodes = graph_data.x.shape[0]
            neg_src = torch.randint(0, num_nodes, (num_neg,))
            neg_dst = torch.randint(0, num_nodes, (num_neg,))
            neg_edge_index = torch.stack([neg_src, neg_dst]).to(self.device)

            # Combined edge labels
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(num_pos),
                torch.zeros(num_neg),
            ]).to(self.device)

            # Forward pass
            pred = self._model(graph_data.x, graph_data.edge_index, edge_label_index)
            loss = F.binary_cross_entropy_with_logits(pred, edge_labels)

            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        return losses

    @torch.no_grad()
    def predict_techniques(
        self,
        graph_data: Data,
        actor_id: str,
        technique_ids: list[str],
        top_k: int = 10,
    ) -> PredictionResult:
        """Predict most likely techniques for a threat actor."""
        if self._model is None:
            raise PredictionError("unknown", "Model not trained. Call train() first.")

        self._model.eval()
        graph_data = graph_data.to(self.device)

        # Get embeddings
        z = self._model.encode(graph_data.x, graph_data.edge_index)

        actor_idx = self._node_mapping.get(actor_id)
        if actor_idx is None:
            raise PredictionError(actor_id, f"Actor not found in graph")

        # Score all technique candidates
        scores: list[tuple[str, float]] = []
        for tech_id in technique_ids:
            tech_idx = self._node_mapping.get(tech_id)
            if tech_idx is None:
                continue

            edge = torch.tensor([[actor_idx], [tech_idx]], dtype=torch.long).to(self.device)
            score = torch.sigmoid(self._model.decode(z, edge)).item()
            scores.append((tech_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_predictions = scores[:top_k]

        predicted = [
            {
                "technique_id": tid,
                "name": self._node_names.get(tid, tid),
                "probability": round(prob, 4),
            }
            for tid, prob in top_predictions
        ]

        actor_embedding = z[actor_idx].cpu().tolist()

        return PredictionResult(
            actor_id=actor_id,
            actor_name=self._node_names.get(actor_id, actor_id),
            predicted_techniques=predicted,
            embedding=actor_embedding,
        )
