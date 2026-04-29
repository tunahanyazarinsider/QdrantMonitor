from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient

BASE_DIR = Path(__file__).resolve().parent


# ── Connection state ─────────────────────────────────────────────────
class QdrantConnection:
    def __init__(self) -> None:
        self.host: str = "localhost"
        self.port: int = 6333
        self.client: QdrantClient | None = None
        self.connected: bool = False
        self.error: str = ""

    def connect(self, host: str | None = None, port: int | None = None) -> None:
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        try:
            self.client = QdrantClient(
                url=f"http://{self.host}:{self.port}", timeout=5
            )
            self.client.get_collections()
            self.connected = True
            self.error = ""
        except Exception as exc:
            self.connected = False
            self.client = None
            self.error = str(exc)


conn = QdrantConnection()
_graph_cache: dict[str, dict] = {}


def _render(request: Request, template: str, ctx: dict) -> HTMLResponse:
    """Starlette 1.0 TemplateResponse wrapper."""
    return templates.TemplateResponse(request, template, context=ctx)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    conn.connect()
    yield


app = FastAPI(title="Qdrant Monitor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Jinja helpers ────────────────────────────────────────────────────
def _trunc(value: object, max_len: int = 80) -> str:
    s = str(value)
    return s[: max_len] + "\u2026" if len(s) > max_len else s


templates.env.filters["trunc"] = _trunc
templates.env.globals["json_dumps"] = lambda v: json.dumps(
    v, ensure_ascii=False, indent=2, default=str
)


# ── Helpers ──────────────────────────────────────────────────────────
def _vec_configs(info) -> list[dict]:
    vectors = info.config.params.vectors
    configs: list[dict] = []
    if isinstance(vectors, dict):
        for name, params in vectors.items():
            configs.append(
                {
                    "name": name,
                    "size": params.size,
                    "distance": (
                        params.distance.value
                        if hasattr(params.distance, "value")
                        else str(params.distance)
                    ),
                }
            )
    elif vectors is not None:
        configs.append(
            {
                "name": "",
                "size": vectors.size,
                "distance": (
                    vectors.distance.value
                    if hasattr(vectors.distance, "value")
                    else str(vectors.distance)
                ),
            }
        )
    return configs


def _sparse_configs(info) -> list[dict]:
    sparse = info.config.params.sparse_vectors
    if sparse and isinstance(sparse, dict):
        return [{"name": n} for n in sparse]
    return []


def _optimizer_status(info) -> str:
    opt = info.optimizer_status
    if hasattr(opt, "status"):
        return str(opt.status)
    if hasattr(opt, "value"):
        return str(opt.value)
    return str(opt)


def _collection_summary(name: str) -> dict:
    info = conn.client.get_collection(name)
    pts = info.points_count or 0
    indexed = getattr(info, "indexed_vectors_count", 0) or 0
    # indexed can exceed points (stale vectors from deletions, or multi-vector)
    index_pct = min(round(indexed / pts * 100, 1), 100.0) if pts > 0 else 0

    # HNSW config
    hnsw = info.config.hnsw_config
    hnsw_cfg = {}
    if hnsw:
        for k in ("m", "ef_construct", "full_scan_threshold", "payload_m", "on_disk", "max_indexing_threads"):
            v = getattr(hnsw, k, None)
            if v is not None:
                hnsw_cfg[k] = v

    # Optimizer config
    opt_cfg = {}
    oc = info.config.optimizer_config
    if oc:
        for k in ("deleted_threshold", "vacuum_min_vector_number", "indexing_threshold", "flush_interval_sec"):
            v = getattr(oc, k, None)
            if v is not None:
                opt_cfg[k] = v

    # WAL config
    wal_cfg = {}
    wc = info.config.wal_config
    if wc:
        for k in ("wal_capacity_mb", "wal_segments_ahead"):
            v = getattr(wc, k, None)
            if v is not None:
                wal_cfg[k] = v

    # Payload schema
    payload_schema = []
    for fname, finfo in (info.payload_schema or {}).items():
        dtype = getattr(finfo, "data_type", None)
        dtype_str = dtype.value if hasattr(dtype, "value") else str(dtype)
        pts_indexed = getattr(finfo, "points", 0) or 0
        payload_schema.append({"field": fname, "type": dtype_str, "points": pts_indexed})

    # Shard / replication
    params = info.config.params
    shard_number = getattr(params, "shard_number", 1) or 1
    replication_factor = getattr(params, "replication_factor", 1) or 1
    on_disk_payload = getattr(params, "on_disk_payload", False)

    return {
        "name": name,
        "points_count": pts,
        "indexed_vectors_count": indexed,
        "index_pct": index_pct,
        "segments_count": info.segments_count or 0,
        "status": (
            info.status.value
            if hasattr(info.status, "value")
            else str(info.status)
        ),
        "optimizer_status": _optimizer_status(info),
        "vec_configs": _vec_configs(info),
        "sparse_configs": _sparse_configs(info),
        "hnsw_config": hnsw_cfg,
        "optimizer_config": opt_cfg,
        "wal_config": wal_cfg,
        "payload_schema": payload_schema,
        "shard_number": shard_number,
        "replication_factor": replication_factor,
        "on_disk_payload": on_disk_payload,
    }


def _payload_fields(collection_name: str) -> list[str]:
    points, _ = conn.client.scroll(collection_name, limit=1, with_payload=True)
    if points and points[0].payload:
        return sorted(points[0].payload.keys())
    return []


def _parse_id(raw: str):
    try:
        return int(raw)
    except ValueError:
        return raw


# ── Routes ───────────────────────────────────────────────────────────
@app.post("/api/connect")
async def api_connect(
    host: str = Form("localhost"),
    port: int = Form(6333),
):
    conn.connect(host, port)
    return RedirectResponse(url="/", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    collections: list[dict] = []
    total_points = 0
    if conn.connected:
        try:
            for c in conn.client.get_collections().collections:
                s = _collection_summary(c.name)
                collections.append(s)
                total_points += s["points_count"]
        except Exception:
            conn.connected = False
    return _render(request, "dashboard.html", {
        "collections": collections,
        "total_points": total_points,
        "conn": conn,
    })


@app.get("/collection/{name}", response_class=HTMLResponse)
async def collection_detail(request: Request, name: str):
    if not conn.connected:
        return RedirectResponse(url="/")
    try:
        summary = _collection_summary(name)
        fields = _payload_fields(name)
    except Exception:
        return RedirectResponse(url="/")
    return _render(request, "collection.html", {
        "collection": summary,
        "payload_fields": fields,
        "conn": conn,
    })


@app.get("/api/{name}/browse", response_class=HTMLResponse)
async def browse_points(
    request: Request,
    name: str,
    limit: int = Query(20, ge=1, le=100),
    offset: str | None = Query(None),
    filter_field: str = Query(""),
    filter_value: str = Query(""),
):
    if not conn.connected:
        return HTMLResponse("<p class='text-red-400 p-4'>Not connected</p>")

    scroll_filter = None
    if filter_field and filter_value:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key=filter_field, match=MatchValue(value=filter_value)
                )
            ]
        )

    parsed_offset = None
    if offset and offset not in ("None", ""):
        parsed_offset = _parse_id(offset)

    points, next_page = conn.client.scroll(
        collection_name=name,
        limit=limit,
        offset=parsed_offset,
        scroll_filter=scroll_filter,
        with_payload=True,
        with_vectors=False,
    )

    all_keys: set[str] = set()
    for p in points:
        if p.payload:
            all_keys.update(p.payload.keys())

    rows = []
    for p in points:
        rows.append(
            {
                "id": p.id,
                "payload": p.payload or {},
                "payload_json": json.dumps(
                    p.payload, ensure_ascii=False, indent=2, default=str
                ),
            }
        )

    return _render(request, "partials/browse.html", {
        "collection_name": name,
        "rows": rows,
        "columns": sorted(all_keys),
        "next_offset": next_page,
        "current_offset": offset,
        "limit": limit,
        "filter_field": filter_field,
        "filter_value": filter_value,
    })


@app.get("/api/{name}/search-id", response_class=HTMLResponse)
async def search_by_id(
    request: Request,
    name: str,
    point_id: str = Query(""),
    with_vectors: bool = Query(False),
):
    if not conn.connected or not point_id.strip():
        return _render(request, "partials/search_id_result.html", {
            "point": None,
            "error": "Not connected" if point_id.strip() else "",
        })

    parsed = _parse_id(point_id.strip())
    try:
        points = conn.client.retrieve(
            collection_name=name,
            ids=[parsed],
            with_payload=True,
            with_vectors=with_vectors,
        )
        if points:
            pt = points[0]
            return _render(request, "partials/search_id_result.html", {
                "point": {
                    "id": pt.id,
                    "payload": pt.payload,
                    "payload_json": json.dumps(
                        pt.payload, ensure_ascii=False, indent=2, default=str
                    ),
                    "vector": pt.vector if with_vectors else None,
                    "vector_json": (
                        json.dumps(pt.vector, default=str, indent=2)
                        if with_vectors and pt.vector
                        else None
                    ),
                },
                "error": "",
            })
        return _render(request, "partials/search_id_result.html", {
            "point": None,
            "error": f"Point \u2018{point_id}\u2019 not found",
        })
    except Exception as exc:
        return _render(request, "partials/search_id_result.html", {
            "point": None, "error": str(exc),
        })


@app.get("/api/{name}/search-text", response_class=HTMLResponse)
async def search_by_text(
    request: Request,
    name: str,
    query: str = Query(""),
    field: str = Query("text"),
    limit: int = Query(10, ge=1, le=50),
):
    if not conn.connected or not query.strip():
        return _render(request, "partials/search_text_result.html", {
            "results": [],
            "query": query,
            "field": field,
            "total_scanned": 0,
            "error": "Not connected" if query.strip() else "",
        })

    query_lower = query.strip().lower()
    matches: list[dict] = []
    next_off = None
    scanned = 0
    max_scroll = 500

    while len(matches) < limit and scanned < max_scroll:
        batch, next_off = conn.client.scroll(
            collection_name=name,
            limit=min(100, max_scroll - scanned),
            offset=next_off,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for p in batch:
            if p.payload and field in p.payload:
                val = str(p.payload[field])
                if query_lower in val.lower():
                    matches.append(
                        {
                            "id": p.id,
                            "payload": p.payload,
                            "payload_json": json.dumps(
                                p.payload,
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ),
                            "matched_value": val,
                        }
                    )
                    if len(matches) >= limit:
                        break
        scanned += len(batch)
        if next_off is None:
            break

    return _render(request, "partials/search_text_result.html", {
        "results": matches,
        "query": query,
        "field": field,
        "total_scanned": scanned,
        "error": "",
    })


@app.get("/api/{name}/search-similar", response_class=HTMLResponse)
async def search_similar(
    request: Request,
    name: str,
    point_id: str = Query(""),
    limit: int = Query(10, ge=1, le=50),
    vector_name: str = Query(""),
):
    if not conn.connected or not point_id.strip():
        return _render(request, "partials/search_similar_result.html", {
            "results": [],
            "error": "Not connected" if not conn.connected else "",
            "query_id": point_id,
        })
    parsed = _parse_id(point_id.strip())
    try:
        results = conn.client.query_points(
            collection_name=name,
            query=parsed,
            using=vector_name if vector_name else None,
            limit=limit,
            with_payload=True,
        )
        rows = [
            {
                "id": pt.id,
                "score": round(pt.score, 6),
                "payload": pt.payload or {},
                "payload_json": json.dumps(
                    pt.payload, ensure_ascii=False, indent=2, default=str
                ),
            }
            for pt in results.points
        ]
        return _render(request, "partials/search_similar_result.html", {
            "results": rows,
            "error": "",
            "query_id": point_id,
        })
    except Exception as exc:
        return _render(request, "partials/search_similar_result.html", {
            "results": [],
            "error": str(exc),
            "query_id": point_id,
        })


def _compute_dist_matrix(vectors: np.ndarray, distance: str) -> np.ndarray:
    if distance == "Cosine":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        v = vectors / norms
        mat = 1.0 - (v @ v.T)
    elif distance == "Dot":
        mat = -(vectors @ vectors.T).astype(float)
    else:  # Euclid
        diff = vectors[:, None, :] - vectors[None, :, :]
        mat = np.sum(diff ** 2, axis=-1).astype(float)
    np.fill_diagonal(mat, np.inf)
    return mat


def _build_hnsw(
    ids: list[str],
    dist_matrix: np.ndarray,
    m: int,
    max_layers: int = 4,
) -> tuple[list[dict], list[dict]]:
    n = len(ids)
    ml = 1.0 / np.log(max(m, 2))
    levels = np.zeros(n, dtype=int)
    for i in range(n):
        while np.random.random() < ml and levels[i] < max_layers:
            levels[i] += 1
    entry = int(np.argmax(levels))
    nodes = [{"id": ids[i], "layer": int(levels[i]), "is_entry": i == entry} for i in range(n)]
    links: list[dict] = []
    seen: set[tuple] = set()
    for layer in range(max_layers + 1):
        mask = np.where(levels >= layer)[0]
        if len(mask) < 2:
            continue
        sub = dist_matrix[np.ix_(mask, mask)]
        k = min(m, len(mask) - 1)
        for li in range(len(mask)):
            gi = mask[li]
            top_k = np.argpartition(sub[li], k)[:k]
            for lj in top_k:
                gj = mask[lj]
                src, tgt = ids[gi], ids[gj]
                edge = (min(src, tgt), max(src, tgt), layer)
                if edge not in seen:
                    seen.add(edge)
                    raw = float(dist_matrix[gi, gj])
                    score = round(1.0 - raw, 4) if raw != np.inf else 0.0
                    links.append({"source": src, "target": tgt, "layer": layer, "score": score})
    return nodes, links


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Reduce to 2D via PCA (numpy SVD, no sklearn needed)."""
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:2].T).astype(float)


@app.get("/api/{name}/vectors-2d")
def vectors_2d(
    name: str,
    vector_name: str = Query(""),
    sample: int = Query(300, ge=50, le=2000),
    color_field: str = Query(""),
):
    if not conn.connected:
        return JSONResponse({"error": "Not connected"}, status_code=400)

    try:
        all_points = []
        next_off = None
        while len(all_points) < sample:
            batch, next_off = conn.client.scroll(
                collection_name=name,
                limit=min(100, sample - len(all_points)),
                offset=next_off,
                with_payload=True,
                with_vectors=True if vector_name == "" else [vector_name],
            )
            if not batch:
                break
            all_points.extend(batch)
            if next_off is None:
                break

        if len(all_points) < 2:
            return JSONResponse({"error": "Not enough points"}, status_code=400)

        # Extract vectors
        vectors = []
        ids = []
        labels = []
        categories = []
        for p in all_points:
            vec = p.vector
            if isinstance(vec, dict):
                vec = vec.get(vector_name)
            if vec is None:
                continue
            vectors.append(vec)
            ids.append(str(p.id))
            # Build hover label from payload
            pay = p.payload or {}
            snippet_fields = ["title", "snippet", "text", "name", "doc_type"]
            lbl_parts = [f"ID: {p.id}"]
            for sf in snippet_fields:
                if sf in pay:
                    val = str(pay[sf])[:80]
                    lbl_parts.append(f"{sf}: {val}")
                    break
            labels.append("<br>".join(lbl_parts))
            cat_val = pay.get(color_field) if color_field else None
            if isinstance(cat_val, list):
                cat_val = cat_val[0] if cat_val else None
            if cat_val is None or cat_val == "":
                categories.append("(no value)")
            else:
                categories.append(str(cat_val))

        mat = np.array(vectors, dtype=np.float32)
        coords = _pca_2d(mat)

        return JSONResponse({
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "ids": ids,
            "labels": labels,
            "categories": categories,
            "count": len(ids),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def _payload_val(payload: dict | None, field: str) -> str:
    if not payload or not field:
        return ""
    v = payload.get(field)
    if isinstance(v, list):
        v = v[0] if v else None
    return str(v) if v is not None else "(no value)"


@app.get("/api/{name}/knn-graph")
def knn_graph(
    name: str,
    sample: int = Query(80, ge=20, le=2000),
    color_field: str = Query(""),
    refresh: bool = Query(False),
):
    """Build HNSW graph from actual vectors using collection's index config."""
    cache_key = f"{name}:{sample}:{color_field}"
    if not refresh and cache_key in _graph_cache:
        return JSONResponse(_graph_cache[cache_key])

    if not conn.connected:
        return JSONResponse({"error": "Not connected"}, status_code=400)

    try:
        info = conn.client.get_collection(name)
        actual_points = info.points_count or 0
        hnsw = info.config.hnsw_config
        m_raw = hnsw.m if hnsw else 16
        ef_construct = getattr(hnsw, "ef_construct", 100) if hnsw else 100
        payload_m = getattr(hnsw, "payload_m", 0) or 0
        # m=0 means no global HNSW index; fall back to payload_m or 16
        effective_m = m_raw if m_raw > 0 else (payload_m if payload_m > 0 else 16)

        # Distance metric + first DENSE vector name (skip sparse)
        vecs_cfg = info.config.params.vectors
        if isinstance(vecs_cfg, dict):
            vec_name = None
            vc = None
            for vn, vcfg in vecs_cfg.items():
                if hasattr(vcfg, "size") and vcfg.size:
                    vec_name = vn
                    vc = vcfg
                    break
            if vec_name is None:
                vec_name = next(iter(vecs_cfg))
                vc = vecs_cfg[vec_name]
        else:
            vec_name = None
            vc = vecs_cfg
        distance_str = (vc.distance.value if hasattr(vc.distance, "value") else str(vc.distance)) if vc else "Cosine"

        # Keyword field for payload_m subgraph grouping
        group_field = ""
        if payload_m > 0:
            for fname, finfo in (info.payload_schema or {}).items():
                dtype = getattr(finfo, "data_type", None)
                dtype_str = dtype.value if hasattr(dtype, "value") else str(dtype)
                if dtype_str == "keyword":
                    group_field = fname
                    break

        need_payload = bool(color_field or group_field)
        payload_spec: list[str] | bool = [f for f in [color_field, group_field] if f] or False

        def _scroll_batch(scroll_filter=None, limit=100, offset=None):
            return conn.client.scroll(
                collection_name=name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=payload_spec if need_payload else False,
                with_vectors=True,
            )

        def _extract_valid(pts, preferred_vec_name):
            """Return (valid_pts, valid_vecs, detected_vec_name)."""
            out_pts, out_vecs = [], []
            detected = None
            for p in pts:
                v = p.vector
                chosen = None
                if isinstance(v, dict):
                    candidates = (
                        ([preferred_vec_name] if preferred_vec_name and preferred_vec_name in v else [])
                        + [vn for vn in v if vn != preferred_vec_name]
                    )
                    for cname in candidates:
                        val = v[cname]
                        if isinstance(val, list) and len(val) > 0:
                            chosen = val
                            if detected is None:
                                detected = cname
                            break
                elif isinstance(v, list) and len(v) > 0:
                    chosen = v
                if chosen is not None:
                    out_pts.append(p)
                    out_vecs.append(chosen)
            return out_pts, out_vecs, detected

        if payload_m > 0 and group_field:
            # payload_m collection: Qdrant builds a separate HNSW per keyword value.
            # Reproduce this by sampling proportionally per group and building
            # disconnected sub-HNSWs. We discover groups first via a quick probe.
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            # Scroll up to min(1000, actual_points) to discover all unique group values
            probe_limit = min(1000, actual_points)
            probe_pts: list = []
            probe_off = None
            while len(probe_pts) < probe_limit:
                b, probe_off = conn.client.scroll(
                    collection_name=name,
                    limit=min(100, probe_limit - len(probe_pts)),
                    offset=probe_off,
                    with_payload=[group_field],
                    with_vectors=False,
                )
                if not b:
                    break
                probe_pts.extend(b)
                if probe_off is None:
                    break
            group_vals = list({_payload_val(p.payload, group_field) for p in probe_pts})
            group_vals = [v for v in group_vals if v and v != "(no value)"][:30]

            per_group = max(sample // max(len(group_vals), 1), effective_m * 2 + 2)

            sampled_all: list = []
            for gv in group_vals:
                g_batch: list = []
                nxt = None
                while len(g_batch) < per_group:
                    b, nxt = conn.client.scroll(
                        collection_name=name,
                        scroll_filter=Filter(must=[FieldCondition(key=group_field, match=MatchValue(value=gv))]),
                        limit=min(50, per_group - len(g_batch)),
                        offset=nxt,
                        with_payload=payload_spec if need_payload else False,
                        with_vectors=True,
                    )
                    if not b:
                        break
                    g_batch.extend(b)
                    if nxt is None:
                        break
                sampled_all.extend(g_batch[:per_group])

            valid_pts, valid_vecs, det = _extract_valid(sampled_all, vec_name)
            if det:
                vec_name = det
                if isinstance(vecs_cfg, dict) and det in vecs_cfg:
                    _vc2 = vecs_cfg[det]
                    distance_str = (_vc2.distance.value if hasattr(_vc2.distance, "value") else str(_vc2.distance))

            if len(valid_pts) < 2:
                return JSONResponse({"error": "Not enough points with vectors"}, status_code=400)

            mat = np.array(valid_vecs, dtype=np.float32)
            ids = [str(p.id) for p in valid_pts]
            dist_matrix = _compute_dist_matrix(mat, distance_str)
            id_to_idx = {sid: i for i, sid in enumerate(ids)}

            # Group indices for per-group HNSW
            groups_idx: dict[str, list[int]] = {}
            for i, p in enumerate(valid_pts):
                gv = _payload_val(p.payload, group_field)
                groups_idx.setdefault(gv, []).append(i)

            all_nodes: list[dict] = []
            all_links: list[dict] = []
            for gv, g_idxs in groups_idx.items():
                if len(g_idxs) < 2:
                    all_nodes.append({
                        "id": ids[g_idxs[0]], "layer": 0, "is_entry": False,
                        "group": gv,
                        "color_value": _payload_val(valid_pts[g_idxs[0]].payload, color_field),
                    })
                    continue
                g_dist = dist_matrix[np.ix_(g_idxs, g_idxs)]
                g_ids = [ids[i] for i in g_idxs]
                g_nodes, g_links = _build_hnsw(g_ids, g_dist, effective_m)
                for nd in g_nodes:
                    nd["group"] = gv
                    nd["color_value"] = _payload_val(valid_pts[id_to_idx[nd["id"]]].payload, color_field)
                all_nodes.extend(g_nodes)
                all_links.extend(g_links)

            nodes, links = all_nodes, all_links
            payload_m_mode = True
        else:
            # Global HNSW: scroll sequentially
            sampled: list = []
            nxt = None
            while len(sampled) < sample:
                b, nxt = _scroll_batch(limit=min(100, sample - len(sampled)), offset=nxt)
                if not b:
                    break
                sampled.extend(b)
                if nxt is None:
                    break

            valid_pts, valid_vecs, det = _extract_valid(sampled, vec_name)
            if det:
                vec_name = det
                if isinstance(vecs_cfg, dict) and det in vecs_cfg:
                    _vc2 = vecs_cfg[det]
                    distance_str = (_vc2.distance.value if hasattr(_vc2.distance, "value") else str(_vc2.distance))

            if len(valid_pts) < 2:
                return JSONResponse({"error": "Not enough points with vectors"}, status_code=400)

            mat = np.array(valid_vecs, dtype=np.float32)
            ids = [str(p.id) for p in valid_pts]
            dist_matrix = _compute_dist_matrix(mat, distance_str)
            id_to_idx = {sid: i for i, sid in enumerate(ids)}

            nodes, links = _build_hnsw(ids, dist_matrix, effective_m)
            for nd in nodes:
                i = id_to_idx[nd["id"]]
                nd["group"] = _payload_val(valid_pts[i].payload, group_field)
                nd["color_value"] = _payload_val(valid_pts[i].payload, color_field)
            payload_m_mode = False

        max_layer = max((nd["layer"] for nd in nodes), default=0)

        result = {
            "config": {
                "m": effective_m,
                "ef_construct": ef_construct,
                "payload_m": payload_m,
                "distance": distance_str,
                "actual_points": actual_points,
                "sampled": len(valid_pts),
                "edges": len(links),
                "max_layer": max_layer,
            },
            "color_field": color_field,
            "group_field": group_field,
            "payload_m_mode": payload_m_mode,
            "nodes": nodes,
            "links": links,
        }
        _graph_cache[cache_key] = result
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/{name}/hnsw-graph")
def hnsw_graph(
    name: str,
    max_nodes: int = Query(200, ge=20, le=500),
):
    """Generate a synthetic HNSW-like graph based on collection index config."""
    if not conn.connected:
        return JSONResponse({"error": "Not connected"}, status_code=400)

    try:
        info = conn.client.get_collection(name)
        actual_points = info.points_count or 0
        hnsw = info.config.hnsw_config
        m = hnsw.m if hnsw else 16
        ef_construct = getattr(hnsw, "ef_construct", 100) if hnsw else 100
        payload_m = getattr(hnsw, "payload_m", None) or 0

        # Payload indexes
        payload_indexes = []
        schema = info.payload_schema or {}
        for fname, finfo in schema.items():
            dtype = getattr(finfo, "data_type", None)
            dtype_str = dtype.value if hasattr(dtype, "value") else str(dtype)
            payload_indexes.append({"field": fname, "type": dtype_str})

        # Determine subgraph mode
        use_payload_subgraphs = m == 0 and payload_m > 0
        effective_m = payload_m if use_payload_subgraphs else m
        if effective_m == 0:
            effective_m = 16

        subgraphs: list[dict] = []
        if use_payload_subgraphs and payload_indexes:
            # Sample unique values for first keyword-indexed field
            group_field = payload_indexes[0]["field"]
            values: dict[str, int] = {}
            next_off = None
            sampled = 0
            while sampled < min(actual_points, 1000):
                batch, next_off = conn.client.scroll(
                    name, limit=100, offset=next_off,
                    with_payload=[group_field], with_vectors=False,
                )
                if not batch:
                    break
                for p in batch:
                    if p.payload and group_field in p.payload:
                        v = str(p.payload[group_field])
                        values[v] = values.get(v, 0) + 1
                sampled += len(batch)
                if next_off is None:
                    break
            for val, cnt in sorted(values.items(), key=lambda x: -x[1]):
                subgraphs.append({"name": val, "count": cnt})
        else:
            subgraphs.append({"name": "global", "count": actual_points})

        # Estimate HNSW layers
        n_layers = max(1, int(np.log(max(actual_points, 2)) / np.log(max(effective_m, 2))))
        n_layers = min(n_layers, 5)

        # Generate synthetic graph
        nodes: list[dict] = []
        links: list[dict] = []
        node_id = 0

        for sg_idx, sg in enumerate(subgraphs):
            sg_n = min(sg["count"], max(max_nodes // len(subgraphs), 10))
            sg_n = max(sg_n, 5)

            # Position subgraphs in a circle layout
            angle = 2 * np.pi * sg_idx / max(len(subgraphs), 1)
            spread = 4.0 if len(subgraphs) > 1 else 0.0
            cx = np.cos(angle) * spread
            cy = np.sin(angle) * spread
            pos = np.random.randn(sg_n, 2) * 1.2 + np.array([cx, cy])

            # Layer assignment (geometric distribution)
            ml = 1.0 / np.log(max(effective_m, 2))
            layers = np.zeros(sg_n, dtype=int)
            for i in range(sg_n):
                while np.random.random() < ml and layers[i] < n_layers - 1:
                    layers[i] += 1

            ids = list(range(node_id, node_id + sg_n))
            for i in range(sg_n):
                nodes.append({
                    "id": ids[i],
                    "subgraph": sg["name"],
                    "layer": int(layers[i]),
                    "x": float(pos[i, 0]) * 60,
                    "y": float(pos[i, 1]) * 60,
                })

            # Build KNN edges per HNSW layer
            for layer in range(n_layers):
                mask = layers >= layer
                local_idx = np.where(mask)[0]
                if len(local_idx) < 2:
                    continue
                layer_pos = pos[local_idx]
                diff = layer_pos[:, None] - layer_pos[None, :]
                dists = np.sum(diff ** 2, axis=-1)
                np.fill_diagonal(dists, np.inf)
                k = min(effective_m, len(local_idx) - 1)
                seen: set[tuple] = set()
                for i in range(len(local_idx)):
                    nbrs = np.argpartition(dists[i], k)[:k]
                    for j in nbrs:
                        src, tgt = ids[int(local_idx[i])], ids[int(local_idx[j])]
                        edge = (min(src, tgt), max(src, tgt))
                        if edge not in seen:
                            seen.add(edge)
                            links.append({"source": src, "target": tgt, "layer": layer})

            node_id += sg_n

        return JSONResponse({
            "config": {
                "m": m, "ef_construct": ef_construct, "payload_m": payload_m,
                "actual_points": actual_points, "layers": n_layers,
            },
            "payload_indexes": payload_indexes,
            "subgraphs": [{"name": s["name"], "count": s["count"]} for s in subgraphs],
            "nodes": nodes,
            "links": links,
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── Entry point ──────────────────────────────────────────────────────
def run():
    import sys

    import uvicorn

    # Ensure the project root is on sys.path so uvicorn's reloader can
    # import "app.main" when spawning a subprocess.
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    uvicorn.run("app.main:app", host="0.0.0.0", port=11000, reload=True)


if __name__ == "__main__":
    run()
