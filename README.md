# Qdrant Monitor

A local dashboard for monitoring and querying Qdrant vector database collections. Built with FastAPI, HTMX, D3.js, and Plotly.

<img width="1452" height="800" alt="Ekran Resmi 2026-04-13 16 05 24" src="https://github.com/user-attachments/assets/280c62fc-1cce-4e48-a487-a318df42a77f" />

<img width="1472" height="774" alt="Ekran Resmi 2026-04-13 16 05 38" src="https://github.com/user-attachments/assets/dcf6bd8a-063f-41dd-b73b-ec0108620f73" />

<img width="1465" height="829" alt="Ekran Resmi 2026-04-13 16 06 08" src="https://github.com/user-attachments/assets/554ad1b5-6bbe-4b67-af71-7506ad135104" />

## Features

- **Dashboard** — Overview of all collections with point counts, indexing coverage, vector configs, status badges
- **Collection Detail** — Full config inspection (HNSW, optimizer, WAL, payload indexes, shard info)
- **Browse Points** — Paginated table with payload filters and expandable JSON detail
- **Search by ID** — Fetch any point by integer or UUID ID
- **Search by Text** — Substring search across any payload field
- **Vector Space Visualization**
  - **HNSW Graph** — Approximate force-directed graph of the HNSW index structure (D3.js)
  - **PCA Projection** — 2D scatter plot of actual vectors colored by payload field (Plotly)

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A running Qdrant instance (default: `localhost:6333`)

## Setup

```bash
# Clone / navigate to the project
cd QdrantMonitor

# Install dependencies
uv sync

# Start the dashboard
uv run python -m app.main
```

The app starts at **http://localhost:11000**.

Enter your Qdrant host and port in the top bar and click **Connect** (defaults to `localhost:6333`).

## Running Qdrant locally

If you don't have Qdrant running yet:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## Tech Stack

- **Backend**: FastAPI + Jinja2 + qdrant-client
- **Frontend**: Tailwind CSS (CDN) + HTMX + D3.js + Plotly.js — no build step
- **Package manager**: uv
