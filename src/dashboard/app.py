"""Streamlit dashboard for the Threat Intelligence Knowledge Graph."""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Threat Intel Knowledge Graph", page_icon="🕸️", layout="wide")

# Entity type color mapping
COLORS = {
    "ThreatActor": "#e74c3c",
    "Technique": "#3498db",
    "Malware": "#9b59b6",
    "Indicator": "#2ecc71",
    "Campaign": "#f39c12",
    "Vulnerability": "#e67e22",
}


def main() -> None:
    st.title("🕸️ Threat Intel Knowledge Graph")
    st.markdown("Real-time threat intelligence with GNN-powered technique prediction")

    tabs = st.tabs(["Graph Explorer", "Feed Ingestion", "Technique Prediction", "Trend Analysis"])

    with tabs[0]:
        render_graph_explorer()
    with tabs[1]:
        render_ingestion_tab()
    with tabs[2]:
        render_prediction_tab()
    with tabs[3]:
        render_trend_tab()


def render_graph_explorer() -> None:
    st.header("Knowledge Graph Explorer")

    col1, col2 = st.columns([1, 3])
    with col1:
        query_type = st.selectbox(
            "Query Type",
            ["full_graph", "actor_techniques", "technique_actors"],
        )
        actor_name = None
        technique_id = None

        if query_type == "actor_techniques":
            actor_name = st.text_input("Actor Name", "APT28")
        elif query_type == "technique_actors":
            technique_id = st.text_input("Technique ID (MITRE)", "T1566")

        limit = st.slider("Node Limit", 10, 500, 100)
        load_graph = st.button("Load Graph", type="primary")

    with col2:
        if load_graph or st.session_state.get("graph_data"):
            if load_graph:
                try:
                    resp = requests.post(
                        f"{API_BASE}/graph/query",
                        json={
                            "query_type": query_type,
                            "actor_name": actor_name,
                            "technique_id": technique_id,
                            "limit": limit,
                        },
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.session_state["graph_data"] = resp.json()
                    else:
                        st.warning("API error. Using demo data.")
                        st.session_state["graph_data"] = _demo_graph()
                except requests.ConnectionError:
                    st.warning("API offline. Loading demo graph.")
                    st.session_state["graph_data"] = _demo_graph()

            data = st.session_state.get("graph_data", _demo_graph())
            render_pyvis_graph(data)


def render_pyvis_graph(data: dict[str, Any]) -> None:
    """Render an interactive graph using pyvis."""
    try:
        from pyvis.network import Network

        net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

        nodes = data.get("nodes", data.get("results", []))
        edges = data.get("edges", [])

        for node in nodes:
            node_id = node.get("id", node.get("stix_id", ""))
            name = node.get("name", node_id)
            node_type = node.get("type", "Entity")
            color = COLORS.get(node_type, "#95a5a6")
            net.add_node(node_id, label=name, color=color, title=f"{node_type}: {name}")

        for edge in edges:
            net.add_edge(
                edge.get("source", ""),
                edge.get("target", ""),
                title=edge.get("rel_type", "RELATED"),
                color="#555",
            )

        html = net.generate_html()
        components.html(html, height=620, scrolling=True)

    except ImportError:
        st.warning("pyvis not available. Showing raw data.")
        st.json(data)


def render_ingestion_tab() -> None:
    st.header("Feed Ingestion")

    col1, col2 = st.columns(2)
    with col1:
        feed = st.selectbox("Feed Source", ["cisa_kev", "otx", "misp"])
        api_key = st.text_input("API Key (if required)", type="password") if feed != "cisa_kev" else None

    with col2:
        st.markdown("### Feed Status")
        try:
            resp = requests.get(f"{API_BASE}/stats", timeout=10)
            if resp.status_code == 200:
                stats = resp.json()
                for key, val in stats.items():
                    st.metric(key.replace("_", " ").title(), val)
        except Exception:
            st.info("Connect to API to view stats.")

    if st.button("Start Ingestion", type="primary"):
        with st.spinner(f"Ingesting from {feed}..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    json={"feed": feed, "api_key": api_key},
                    timeout=120,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result["success"]:
                        st.success(
                            f"Ingested {result['entities_ingested']} entities, "
                            f"{result['relationships_ingested']} relationships"
                        )
                    else:
                        st.error(f"Ingestion failed: {result.get('error')}")
            except requests.ConnectionError:
                st.error("Cannot connect to API.")


def render_prediction_tab() -> None:
    st.header("GNN Technique Prediction")
    st.markdown("Predict which ATT&CK techniques a threat actor is likely to adopt next.")

    actor_id = st.text_input("Threat Actor ID", placeholder="threat-actor--abc123")
    top_k = st.slider("Number of Predictions", 3, 20, 10)

    if st.button("Predict", type="primary") and actor_id:
        with st.spinner("Training GNN and generating predictions..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/predict",
                    json={"actor_id": actor_id, "top_k": top_k},
                    timeout=120,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.subheader(f"Predictions for: {result['actor_name']}")
                    _render_predictions(result["predicted_techniques"])
                else:
                    st.error(resp.text)
            except requests.ConnectionError:
                st.warning("API offline. Showing demo predictions.")
                _render_predictions(_demo_predictions())

    if st.checkbox("Show Demo Predictions"):
        _render_predictions(_demo_predictions())


def _render_predictions(techniques: list[dict[str, Any]]) -> None:
    if not techniques:
        st.info("No predictions available.")
        return

    names = [t["name"] for t in techniques]
    probs = [t["probability"] for t in techniques]

    fig = go.Figure(go.Bar(
        x=probs,
        y=names,
        orientation="h",
        marker_color=["#e74c3c" if p > 0.7 else "#f39c12" if p > 0.4 else "#3498db" for p in probs],
    ))
    fig.update_layout(
        title="Predicted Technique Adoption Probability",
        xaxis_title="Probability",
        yaxis=dict(autorange="reversed"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trend_tab() -> None:
    st.header("Temporal Trend Analysis")

    # Demo trend visualization
    import numpy as np

    techniques = ["T1566 (Phishing)", "T1059 (Command Line)", "T1078 (Valid Accounts)",
                  "T1190 (Exploit Public)", "T1053 (Scheduled Task)"]
    days = [f"Day {i+1}" for i in range(30)]

    fig = go.Figure()
    np.random.seed(42)
    for tech in techniques:
        base = np.random.uniform(2, 8)
        trend = np.random.uniform(-0.1, 0.2)
        values = [max(0, base + trend * i + np.random.normal(0, 1.5)) for i in range(30)]
        fig.add_trace(go.Scatter(x=days, y=values, name=tech, mode="lines"))

    fig.update_layout(
        title="Technique Observation Frequency (30-Day Window)",
        xaxis_title="Day",
        yaxis_title="Observations",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔺 Emerging Techniques")
        st.markdown("- **T1566**: Phishing (z-score: 3.2, EMERGING)")
        st.markdown("- **T1078**: Valid Accounts (z-score: 2.5, EMERGING)")
    with col2:
        st.subheader("🔻 Declining Techniques")
        st.markdown("- **T1053**: Scheduled Task (slope: -0.15)")


def _demo_graph() -> dict[str, Any]:
    nodes = [
        {"id": "actor-1", "name": "APT28", "type": "ThreatActor"},
        {"id": "actor-2", "name": "Lazarus Group", "type": "ThreatActor"},
        {"id": "tech-1", "name": "T1566 Phishing", "type": "Technique"},
        {"id": "tech-2", "name": "T1059 Command Line", "type": "Technique"},
        {"id": "tech-3", "name": "T1078 Valid Accounts", "type": "Technique"},
        {"id": "mal-1", "name": "Emotet", "type": "Malware"},
        {"id": "mal-2", "name": "TrickBot", "type": "Malware"},
        {"id": "ioc-1", "name": "192.168.1.100", "type": "Indicator"},
        {"id": "camp-1", "name": "Operation Fancy Bear", "type": "Campaign"},
    ]
    edges = [
        {"source": "actor-1", "target": "tech-1", "rel_type": "USES"},
        {"source": "actor-1", "target": "tech-2", "rel_type": "USES"},
        {"source": "actor-2", "target": "tech-3", "rel_type": "USES"},
        {"source": "actor-1", "target": "mal-1", "rel_type": "DEPLOYS"},
        {"source": "actor-2", "target": "mal-2", "rel_type": "DEPLOYS"},
        {"source": "mal-1", "target": "ioc-1", "rel_type": "INDICATES"},
        {"source": "camp-1", "target": "actor-1", "rel_type": "ATTRIBUTED_TO"},
        {"source": "camp-1", "target": "tech-1", "rel_type": "USES"},
    ]
    return {"nodes": nodes, "edges": edges}


def _demo_predictions() -> list[dict[str, Any]]:
    return [
        {"technique_id": "T1566", "name": "Phishing", "probability": 0.89},
        {"technique_id": "T1059", "name": "Command and Scripting Interpreter", "probability": 0.76},
        {"technique_id": "T1078", "name": "Valid Accounts", "probability": 0.68},
        {"technique_id": "T1190", "name": "Exploit Public-Facing Application", "probability": 0.55},
        {"technique_id": "T1053", "name": "Scheduled Task/Job", "probability": 0.42},
    ]


if __name__ == "__main__":
    main()
