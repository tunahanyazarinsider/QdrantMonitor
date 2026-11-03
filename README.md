# Qdrant Monitor

A local dashboard for monitoring Qdrant clusters and for browsing, visualizing and managing collections. Built with FastAPI, HTMX, D3.js, and Plotly — no build step required.

<img width="1452" height="800" alt="Dashboard" src="https://github.com/user-attachments/assets/280c62fc-1cce-4e48-a487-a318df42a77f" />

<img width="1472" height="774" alt="Collection detail" src="https://github.com/user-attachments/assets/dcf6bd8a-063f-41dd-b73b-ec0108620f73" />

<img width="1465" height="829" alt="Browse points" src="https://github.com/user-attachments/assets/554ad1b5-6bbe-4b67-af71-7506ad135104" />

**HNSW Graph** — real index structure, layer-stratified, colored by payload field:

<img width="1465" alt="HNSW Graph visualization" src="docs/hnsw_graph.png" />

## Features

### Cluster page (`/cluster`)
- **Health strip** — leader, `healthz` / `readyz`, open issues, consensus thread, recovery mode, quotas, version skew, raft agreement across peers, replica errors
- **Resources** — RAM, disk, CPU, file descriptors and mmaps against their limits
- **Memory** — allocator breakdown (allocated / active / resident / retained) and measured collection memory from `/collections/{name}/memory`: heap, page cache, expected cache, warm %, and a RAM budget (heap + expected cache vs RAM)
- **Live charts** — memory, requests/s and failures/s, major page faults/s, sampled from `/metrics` every 5 s; backs off when Qdrant is unreachable
- **Consensus & peers** — raft term, commit and pending ops; every peer's own view of role, leader, term and commit; RAM and collection count per peer
- **Collections** — cluster-wide point count, readable replicas, disk / heap / cache warmth, memory tiers, optimizations, CPU and IO counters
- **API latency** — per endpoint count, avg, p95 (from histogram buckets) and max
- **Slow requests** — operations over 50 ms from `/profiler/slow_requests`, with request bodies

### Multi-peer view
Peer URIs reported by `/cluster` are cluster-internal. Map each peer to a URL the app can reach (for example a `kubectl port-forward`) under **Peer endpoints**, or click **Scan** to probe ±20 ports on the connected host. Each URL is checked with `GET /cluster` and kept only if its peer id belongs to the cluster; a URL that now serves another peer is flagged instead of shown under the wrong id.

### Collection health tab
- **Findings** — rule-based checks: replica state disagreement between peers, non-readable replicas (`ManualRecovery`, `Dead`, …), replicas with diverging point counts, failed forwarded updates, payload indexes that cover 0 points (with their disk cost), cold working set, `m=0` full-scan warning, near-vacuum segments
- **Memory** — per component (vectors, HNSW index, quantized, payload, each payload index, ID tracker): tier, disk, heap, page cache, expected cache, warm %
- **Replicas** — every replica's own state and point count read from the peer that holds it, Δ vs the largest replica, how each other peer sees it, and forwarded update/search counters
- **API latency** for this collection, **payload index coverage** with top facet values, **segments**, **search paths** (HNSW vs plain/exact), **optimizer** queue and log, **slow requests**

### Collection management
- Create and delete collections
- Create a collection from an uploaded snapshot
- Migrate points from another Qdrant instance
- Snapshots tab: create, list, download, save to a path, delete

### Collections dashboard
- Overview of all collections — point counts, indexed vectors, indexing %, segment count, shard count
- Status badges (green/yellow/red)
- Vector config chips (name, dimensions, distance metric, sparse)

### Collection detail
- Full config inspection: HNSW params (M, ef_construct, payload_m), optimizer, WAL, quantization
- Payload schema with indexed field types
- Shard and segment breakdown

### Browse & search
- **Browse** — paginated point table with expandable JSON payload
- **By ID** — fetch any point by integer or UUID
- **By Text** — substring search across any payload field
- **Similar** — find nearest neighbors by Point ID using `query_points` API; configurable limit and named vector

### Vector Space visualization

#### HNSW Graph
Real graph built from actual vectors using the collection's index config (M, distance, payload_m):

- Nodes assigned HNSW layers probabilistically (matching Qdrant's `ml = 1/ln(M)` formula)
- **Layer stratification** — L0 at bottom, entry-point layer at top; dashed band lines per layer
- **Pyramid X constraint** — higher layers cluster narrower horizontally
- **Node size scales with layer** — entry point has white ring
- **Layer toggle checkboxes** — show/hide any layer independently
- **Collapse L0** — shrink dense base-layer nodes to dots to reveal higher-layer structure
- **Color by payload field** — color nodes by any keyword/string payload field
- **Node click → similar points** — click any node to fetch its nearest neighbors via Qdrant's `query_points` API
- **payload_m support** — when `payload_m > 0`, builds per-keyword-value sub-HNSWs matching Qdrant's actual index structure (disconnected subgraphs per keyword value)
- Sample up to 2000 points; results cached, refresh on demand
- Zoom, pan, drag nodes

#### PCA Projection
- 2D scatter of actual vectors via PCA (numpy SVD, no sklearn)
- Color by any payload field
- **Click a point** → fetch similar vectors via `query_points`, shown inline below the plot
- Named vector support for multi-vector collections

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A running Qdrant instance (default: `localhost:6333`)

## Setup

```bash
git clone https://github.com/tunahanyazarinsider/QdrantMonitor
cd QdrantMonitor

uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 11000 --reload
```

The app starts at **http://localhost:11000**.

Enter your Qdrant host and port in the top bar and click **Connect** (defaults to `localhost:6333`).

## Version notes

- Qdrant 1.18+ is needed for measured per-component memory; older servers fall back to tiers derived from config.
- Qdrant before 1.19 reports the host machine's RAM and cores in telemetry, not the container limit; the Resources panel labels these as host values.
- `/quotas` exists from 1.19 and `/profiler/slow_requests` from 1.16; panels for missing endpoints stay empty.

## Checks

```bash
python3 app/insights.py   # assert-based self-check of the metrics, memory and replica logic
```

## Running Qdrant locally

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Backend | FastAPI, Jinja2, httpx, qdrant-client ≥ 1.17 |
| Frontend | Tailwind CSS (CDN), HTMX, D3.js v7, Plotly.js |
| Visualization | D3 force simulation, numpy SVD (PCA) |
| Package manager | uv |
