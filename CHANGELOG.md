# Changelog

User-visible changes, newest first. Sections are dated; add new work under **Unreleased** in the same commit as the change, and move it under a dated heading when it is pushed.

## Unreleased

### Changed
- Collection page opens on a new **Overview** tab (formerly Health); Browse is the second tab.

## 2026-11-03

### Added
- **Home page overview**: cluster health strip (leader, healthz/readyz, issues, consensus, recovery mode, quota, versions, raft views, collections, replicas) and one table of every collection: status, optimizer, cluster-wide points and vectors, readable replicas, shards × RF, HNSW-indexed share, segments, disk / heap / cache warmth, error and warning counts.
- **Cluster page** (`/cluster`): resources against limits, allocator memory, measured collection memory per tier with a RAM budget, live charts (memory, requests and failures per second, major page faults) sampled from `/metrics`, consensus and per-peer raft view, API latency per endpoint (avg, p95, max), slow requests from `/profiler/slow_requests`, node details.
- **Replicas** on the cluster page: every collection × every peer with each replica's state, points and status (green / yellow / grey / red), per-shard detail on hover.
- **Collection health** (now Overview tab): findings, measured memory per component (vectors, HNSW index, quantized, payload, each payload index, ID tracker) with tier, disk, heap, page cache, expected cache and warm %; shard × peer **placement** grid; **replicas** table with each replica's own state, points, Δ vs the largest replica, how other peers see it and forwarded update/search counters; payload index coverage with top facet values; segments; search paths; optimizer queue and log; slow requests; per-collection API latency.
- **Segments per peer**: state (plain / indexed, appendable / immutable), HNSW-indexed share, deleted and deferred vectors, vector storage, HNSW parameters, quantization, payload storage, payload index types and coverage per segment, and **config drift** against the collection config.
- **Multi-peer view**: replica state and point counts of every peer through `/cluster/telemetry` with one port-forward; optional per-peer endpoint mapping (manual or port scan) for RAM and slow-request data, verified by peer id.
- Findings for: peers disagreeing on a replica's state, non-readable replicas (`ManualRecovery`, `Dead`, …), missing replicas, diverging point counts, failed forwarded updates, under-replicated or oversized shards, uneven placement, peers holding the only readable copy, non-green collection status, optimizer errors, payload indexes covering 0 points (with their disk cost), cold working set, segments over `indexing_threshold`, unindexed vectors, `m=0` full scans, near-vacuum segments.
- Version notes: Qdrant before 1.19 reports host RAM and cores rather than the container limit, and the UI labels them as host values.
- Collection migrate between Qdrant instances, and create-from-snapshot upload.
- `CLAUDE.md` (run `/code-review` after code changes) and this changelog.

### Fixed
- Escaped user-controlled values in the segments response and the delete confirmation dialog.
- `indexing_threshold` is read as KiB per segment and per vector name.
- Warm % counts each component's cache only up to its expected cache.
- Many crashes on missing or `null` fields across Qdrant 1.17–1.20 responses.

## 2026-05-01

### Added
- Create and delete collections from the dashboard.

## 2026-04-29

### Added
- Vector space: real KNN graph, HNSW graph built from the collection's vectors and config (layer stratification, layer toggles, collapse L0, colour by payload field, click a node for similar points, `payload_m` sub-graphs), PCA projection with click-to-explore, similar-point search.

### Fixed
- `m=0` / `payload_m` collections, sparse vector detection, collections with some empty named vectors.

## 2026-04-13

### Added
- Initial release: collections dashboard, collection detail with config, browse, search by ID and by text.
