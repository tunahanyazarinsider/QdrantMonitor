from __future__ import annotations

import asyncio
import html
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient

from app import insights

BASE_DIR = Path(__file__).resolve().parent
_CONN_STATE_FILE = BASE_DIR.parent / ".qdrant_conn.json"


def _load_conn_state() -> tuple[str, int, dict[str, str]]:
    try:
        data = json.loads(_CONN_STATE_FILE.read_text())
        peers = data.get("peer_urls") if isinstance(data.get("peer_urls"), dict) else {}
        return str(data.get("host", "localhost")), int(data.get("port", 6333)), {str(k): str(v) for k, v in peers.items()}
    except Exception:
        return "localhost", 6333, {}


def _save_conn_state(host: str, port: int, peer_urls: dict[str, str] | None = None) -> None:
    try:
        _CONN_STATE_FILE.write_text(json.dumps({"host": host, "port": port, "peer_urls": peer_urls or {}}))
    except Exception:
        pass


# ── Connection state ─────────────────────────────────────────────────
class QdrantConnection:
    def __init__(self) -> None:
        h, p, peers = _load_conn_state()
        self.host: str = h
        self.port: int = p
        # peer_id -> REST base URL reachable from this app (peer URIs in /cluster are cluster-internal).
        self.peer_urls: dict[str, str] = peers
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
            _save_conn_state(self.host, self.port, self.peer_urls)
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


@app.post("/api/collections/create")
async def api_create_collection(
    name: str = Form(...),
    vector_size: int = Form(...),
    distance: str = Form("Cosine"),
    vector_name: str = Form(""),
    on_disk: bool = Form(False),
):
    if not conn.connected:
        return JSONResponse({"error": "Not connected"}, status_code=400)
    distance = distance.capitalize()
    if distance not in {"Cosine", "Dot", "Euclid", "Manhattan"}:
        return JSONResponse(
            {"error": f"Invalid distance: {distance}"}, status_code=400
        )
    vec_cfg = {"size": vector_size, "distance": distance, "on_disk": on_disk}
    body: dict = {"vectors": {vector_name: vec_cfg} if vector_name else vec_cfg}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.put(
                f"http://{conn.host}:{conn.port}/collections/{name}",
                json=body,
            )
        if resp.status_code >= 400:
            return JSONResponse(
                {"error": f"{resp.status_code}: {resp.text[:300]}"},
                status_code=400,
            )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return RedirectResponse(url="/", status_code=303)


@app.post("/api/collections/migrate")
async def api_migrate_collection(
    source_url: str = Form(...),
    source_collection: str = Form(...),
    target_collection: str = Form(...),
    batch_size: int = Form(256),
    create_target: bool = Form(False),
):
    if not conn.connected:
        return JSONResponse({"error": "Not connected to target"}, status_code=400)
    src = source_url.rstrip("/")
    target = _qdrant_base_url()
    moved = 0
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            info_resp = await client.get(f"{src}/collections/{source_collection}")
            if info_resp.status_code >= 400:
                return JSONResponse(
                    {"error": f"Source info failed: {info_resp.status_code} {info_resp.text[:300]}"},
                    status_code=400,
                )
            src_info = info_resp.json().get("result", {})
            src_cfg = src_info.get("config", {}).get("params", {})

            if create_target:
                exists = await client.get(f"{target}/collections/{target_collection}")
                if exists.status_code == 404:
                    body = {"vectors": src_cfg.get("vectors")}
                    if src_cfg.get("sparse_vectors"):
                        body["sparse_vectors"] = src_cfg["sparse_vectors"]
                    create = await client.put(
                        f"{target}/collections/{target_collection}", json=body
                    )
                    if create.status_code >= 400:
                        return JSONResponse(
                            {"error": f"Create target failed: {create.text[:300]}"},
                            status_code=400,
                        )

            offset = None
            while True:
                scroll_body = {
                    "limit": batch_size,
                    "with_payload": True,
                    "with_vector": True,
                }
                if offset is not None:
                    scroll_body["offset"] = offset
                sc = await client.post(
                    f"{src}/collections/{source_collection}/points/scroll",
                    json=scroll_body,
                )
                if sc.status_code >= 400:
                    return JSONResponse(
                        {"error": f"Scroll failed at moved={moved}: {sc.text[:300]}"},
                        status_code=500,
                    )
                data = sc.json().get("result", {})
                points = data.get("points", []) or []
                if not points:
                    break

                upsert_body = {
                    "points": [
                        {
                            "id": p["id"],
                            "vector": p.get("vector"),
                            "payload": p.get("payload") or {},
                        }
                        for p in points
                    ]
                }
                up = await client.put(
                    f"{target}/collections/{target_collection}/points?wait=true",
                    json=upsert_body,
                )
                if up.status_code >= 400:
                    return JSONResponse(
                        {"error": f"Upsert failed at moved={moved}: {up.text[:400]}"},
                        status_code=500,
                    )
                moved += len(points)
                offset = data.get("next_page_offset")
                if offset is None:
                    break
    except Exception as exc:
        return JSONResponse(
            {"error": f"Migration error at moved={moved}: {exc}"},
            status_code=500,
        )
    return JSONResponse({
        "ok": True,
        "moved": moved,
        "source": f"{src}/{source_collection}",
        "target": f"{target}/{target_collection}",
    })


@app.post("/api/collections/restore-from-snapshot")
async def api_restore_new_collection(
    request: Request,
    name: str = Form(...),
    snapshot_file: UploadFile = File(...),
    priority: str = Form("snapshot"),
):
    if not conn.connected:
        return JSONResponse({"error": "Not connected"}, status_code=400)
    import tempfile

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".snapshot") as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await snapshot_file.read(1 << 20)
                if not chunk:
                    break
                tmp.write(chunk)
        url = (
            f"{_qdrant_base_url()}/collections/{name}"
            f"/snapshots/upload?priority={priority}"
        )
        async with httpx.AsyncClient(timeout=None) as client:
            with tmp_path.open("rb") as f:
                files = {
                    "snapshot": (
                        snapshot_file.filename or "upload.snapshot",
                        f,
                        "application/octet-stream",
                    )
                }
                resp = await client.post(url, files=files)
        if resp.status_code >= 400:
            return JSONResponse(
                {"error": f"{resp.status_code}: {resp.text[:400]}"},
                status_code=400,
            )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return RedirectResponse(url="/", status_code=303)


@app.post("/api/collections/{name}/delete")
async def api_delete_collection(name: str):
    if not conn.connected:
        return JSONResponse({"error": "Not connected"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(
                f"http://{conn.host}:{conn.port}/collections/{name}"
            )
        if resp.status_code >= 400:
            return JSONResponse(
                {"error": f"{resp.status_code}: {resp.text[:300]}"},
                status_code=400,
            )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return RedirectResponse(url="/", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not conn.connected:
        return _render(request, "dashboard.html", {"collections": [], "total_points": 0, "conn": conn, "health": []})
    state = await _cluster_state(request)
    for row in state["collections"]:
        mx = row.get("replicas") or {}
        row["cluster_points"] = sum(sh["max_points"] for sh in mx.get("shards") or [] if sh["max_points"] is not None) \
            if mx.get("shards") else row["points"]
    state["total_points"] = sum(r["cluster_points"] for r in state["collections"])
    return _render(request, "dashboard.html", state)


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
        existing = conn.client.retrieve(
            collection_name=name,
            ids=[parsed],
            with_payload=False,
            with_vectors=True,
        )
        if not existing:
            return _render(request, "partials/search_similar_result.html", {
                "results": [],
                "error": f"Point ID {point_id} not found in collection",
                "query_id": point_id,
            })
        vec = existing[0].vector
        query_vector: list[float] | None = None
        if isinstance(vec, dict):
            if vector_name and vector_name not in vec:
                return _render(request, "partials/search_similar_result.html", {
                    "results": [],
                    "error": f"Point has no vector named '{vector_name}'. Available: {list(vec.keys())}",
                    "query_id": point_id,
                })
            if vector_name and vec.get(vector_name):
                query_vector = list(vec[vector_name])
            else:
                for k, v in vec.items():
                    if v:
                        vector_name = k
                        query_vector = list(v)
                        break
            if query_vector is None:
                return _render(request, "partials/search_similar_result.html", {
                    "results": [],
                    "error": "Point has no populated named vectors",
                    "query_id": point_id,
                })
        elif vec is None:
            return _render(request, "partials/search_similar_result.html", {
                "results": [],
                "error": "Point exists but has no vector data",
                "query_id": point_id,
            })
        else:
            query_vector = list(vec)
        results = conn.client.query_points(
            collection_name=name,
            query=query_vector,
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


# ── Snapshots ────────────────────────────────────────────────────────
def _qdrant_base_url() -> str:
    return f"http://{conn.host}:{conn.port}"


def _snapshot_url(name: str, snap_name: str) -> str:
    return f"{_qdrant_base_url()}/collections/{name}/snapshots/{snap_name}"


async def _list_snaps_raw(name: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{_qdrant_base_url()}/collections/{name}/snapshots")
        resp.raise_for_status()
        data = resp.json()
    rows = []
    for s in data.get("result", []) or []:
        rows.append({
            "name": s.get("name"),
            "size": s.get("size"),
            "creation_time": s.get("creation_time"),
            "checksum": s.get("checksum"),
        })
    rows.sort(key=lambda r: r.get("creation_time") or "", reverse=True)
    return rows


@app.get("/api/{name}/snapshots", response_class=HTMLResponse)
async def list_snapshots(request: Request, name: str):
    if not conn.connected:
        return HTMLResponse("<p class='text-red-400 p-4 text-xs'>Not connected</p>")
    try:
        rows = await _list_snaps_raw(name)
        return _render(request, "partials/snapshots_list.html", {
            "snapshots": rows,
            "collection_name": name,
            "error": "",
        })
    except Exception as exc:
        return _render(request, "partials/snapshots_list.html", {
            "snapshots": [],
            "collection_name": name,
            "error": str(exc),
        })


@app.post("/api/{name}/snapshots/create", response_class=HTMLResponse)
async def create_snapshot(request: Request, name: str):
    if not conn.connected:
        return HTMLResponse("<p class='text-red-400 p-4 text-xs'>Not connected</p>")
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                f"{_qdrant_base_url()}/collections/{name}/snapshots"
            )
        if resp.status_code >= 400:
            return _render(request, "partials/snapshots_list.html", {
                "snapshots": await _list_snaps_raw(name),
                "collection_name": name,
                "error": f"Create failed: {resp.status_code} {resp.text[:300]}",
            })
    except Exception as exc:
        return _render(request, "partials/snapshots_list.html", {
            "snapshots": [],
            "collection_name": name,
            "error": f"Create failed: {exc}",
        })
    return await list_snapshots(request, name)


@app.post("/api/{name}/snapshots/{snap_name}/delete", response_class=HTMLResponse)
async def delete_snapshot(request: Request, name: str, snap_name: str):
    if not conn.connected:
        return HTMLResponse("<p class='text-red-400 p-4 text-xs'>Not connected</p>")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.delete(_snapshot_url(name, snap_name))
        if resp.status_code >= 400:
            return _render(request, "partials/snapshots_list.html", {
                "snapshots": await _list_snaps_raw(name),
                "collection_name": name,
                "error": f"Delete failed: {resp.status_code} {resp.text[:300]}",
            })
    except Exception as exc:
        return _render(request, "partials/snapshots_list.html", {
            "snapshots": [],
            "collection_name": name,
            "error": f"Delete failed: {exc}",
        })
    return await list_snapshots(request, name)


@app.get("/api/{name}/snapshots/{snap_name}/download")
async def download_snapshot(name: str, snap_name: str):
    if not conn.connected:
        return JSONResponse({"error": "Not connected"}, status_code=400)
    url = _snapshot_url(name, snap_name)

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(f"Qdrant {resp.status_code}: {body.decode(errors='replace')}")
                async for chunk in resp.aiter_bytes(chunk_size=1 << 20):
                    yield chunk

    return StreamingResponse(
        stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{snap_name}"'},
    )


@app.post("/api/{name}/snapshots/{snap_name}/save", response_class=HTMLResponse)
async def save_snapshot_to_path(
    request: Request,
    name: str,
    snap_name: str,
    dest_path: str = Form(...),
):
    if not conn.connected:
        return HTMLResponse("<p class='text-red-400 p-2 text-xs'>Not connected</p>")
    dest = Path(dest_path).expanduser()
    if dest.is_dir() or dest_path.endswith("/"):
        dest = dest / snap_name
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return HTMLResponse(
            f"<p class='text-red-400 p-2 text-xs'>Cannot create dir: {exc}</p>"
        )
    url = _snapshot_url(name, snap_name)
    try:
        bytes_written = 0
        with dest.open("wb") as f:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        return HTMLResponse(
                            f"<p class='text-red-400 p-2 text-xs'>Qdrant {resp.status_code}: "
                            f"{body.decode(errors='replace')[:200]}</p>"
                        )
                    async for chunk in resp.aiter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                        bytes_written += len(chunk)
        return HTMLResponse(
            f"<p class='text-[#00e5a0] p-2 text-xs'>Saved {bytes_written:,} bytes → "
            f"<span class='font-mono'>{dest}</span></p>"
        )
    except Exception as exc:
        return HTMLResponse(
            f"<p class='text-red-400 p-2 text-xs'>Save failed: {exc}</p>"
        )


# ── Cluster & collection insights ────────────────────────────────────
def _fmt_bytes(n) -> str:
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        return "—"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


templates.env.filters["bytes"] = _fmt_bytes


async def _qget(client: httpx.AsyncClient, path: str, raw: bool = False,
                base: str | None = None, optional: bool = False):
    """Returns (payload, error). payload is `result` for JSON endpoints, text when raw.
    optional=True treats 404 as "not supported by this Qdrant version" rather than an error."""
    try:
        resp = await client.get(f"{base or _qdrant_base_url()}{path}")
    except httpx.TimeoutException as exc:
        return None, f"{path}: timed out ({type(exc).__name__})"
    except httpx.HTTPError as exc:
        return None, f"{path}: {type(exc).__name__}: {exc}"
    if optional and resp.status_code == 404:
        return None, None
    if raw:
        return resp.text, (None if resp.status_code < 400 else f"{path}: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        return None, f"{path}: HTTP {resp.status_code} {resp.text[:200]}"
    try:
        body = resp.json()
    except ValueError:
        return None, f"{path}: non-JSON response"
    if not isinstance(body, dict):
        return None, f"{path}: unexpected response shape"
    return body.get("result"), None


def _d(v) -> dict:
    return v if isinstance(v, dict) else {}


def _cluster_shards(cluster_telemetry, collection: str) -> list | None:
    """Per-replica state and size of every peer, as gathered by the connected node; None if unavailable."""
    entry = _d(_d(cluster_telemetry).get("collections")).get(collection)
    return _d(entry).get("shards") if isinstance(entry, dict) and "shards" in entry else None


def _normalize_url(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    return raw if raw.startswith(("http://", "https://")) else f"http://{raw}"


async def _peer_bases(client: httpx.AsyncClient) -> tuple[dict, dict[str, str | None]]:
    """Returns (/cluster of the connected node, peer_id -> reachable base URL or None)."""
    clus, _ = await _qget(client, "/cluster")
    clus = _d(clus)
    self_id = str(clus.get("peer_id") or "")
    bases: dict[str, str | None] = {}
    for pid in _d(clus.get("peers")) or ({self_id: {}} if self_id else {}):
        bases[str(pid)] = _qdrant_base_url() if str(pid) == self_id else conn.peer_urls.get(str(pid))
    if not bases:
        # Cluster mode disabled: /cluster has no peer_id, the node is its own only replica holder.
        bases["local"] = _qdrant_base_url()
    return clus, bases


async def _replica_views(client: httpx.AsyncClient, bases: dict[str, str | None], collection: str,
                         telemetry: dict[str, dict] | None = None):
    """Each peer's own view of the collection's shards, plus its per-replica op counters."""
    q = quote(collection, safe="")
    reachable = {pid: b for pid, b in bases.items() if b}
    results = await asyncio.gather(*(_qget(client, f"/collections/{q}/cluster", base=b) for b in reachable.values()))
    views: dict[str, dict | None] = {pid: None for pid in bases}
    errors = []
    for pid, (res, err) in zip(reachable, results):
        got = str(_d(res).get("peer_id") or "")
        if res is not None and pid != "local" and got and got != pid:
            errors.append(f"peer {pid}: endpoint {reachable[pid]} now serves peer {got}; remap it")
            continue
        if pid == "local" and got:
            # Cluster mode disabled: key the only node by the peer id its shards report.
            views.pop("local", None)
            pid = got
        views[pid] = _d(res) if res is not None else None
        if err:
            errors.append(f"peer {pid}: {err}")
    remote_stats, local_status = {}, {}
    for pid, tele in (telemetry or {}).items():
        for c in _d(_d(tele).get("collections")).get("collections") or []:
            if c.get("id") == collection:
                remote_stats[pid] = [r for s in c.get("shards") or [] for r in s.get("remote") or []]
                own = str(pid)
                if own in ("", "local") and len(views) == 1:
                    own = next(iter(views))
                for s in c.get("shards") or []:
                    if _d(s.get("local")).get("status"):
                        local_status[(s.get("id"), own)] = s["local"]["status"]
    return views, remote_stats, errors, local_status


async def _resolve_peer(client: httpx.AsyncClient, url: str) -> str | None:
    res, _ = await _qget(client, "/cluster", base=url)
    pid = _d(res).get("peer_id")
    return str(pid) if pid is not None else None


@app.post("/api/peers/save")
async def api_peers_save(urls: str = Form("")):
    if not conn.connected:
        return RedirectResponse(url="/", status_code=303)
    candidates = [_normalize_url(u) for u in urls.replace(",", "\n").splitlines() if u.strip()]
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
        clus, cerr = await _qget(client, "/cluster")
        known = {str(p) for p in _d(_d(clus).get("peers"))}
        resolved = await asyncio.gather(*(_resolve_peer(client, u) for u in candidates))
    if not known:
        return RedirectResponse(url=f"/cluster?msg={quote('Peer list unavailable; endpoints not changed. ' + (cerr or ''))}",
                                status_code=303)
    mapped, skipped, blips = {}, [], set()
    for url, pid in zip(candidates, resolved):
        if pid and pid in known:
            mapped[pid] = url
        else:
            skipped.append(url)
            if pid is None:
                blips.add(url)
    # Keep a mapping whose URL was submitted but did not answer (port-forward blip); drop everything else
    # that was not re-confirmed, including peers that left the cluster.
    kept = {k: v for k, v in conn.peer_urls.items() if k in known and k not in mapped and v in blips}
    conn.peer_urls = {**kept, **mapped}
    _save_conn_state(conn.host, conn.port, conn.peer_urls)
    msg = f"Mapped {len(mapped)} peer(s)." + (f" Not part of this cluster or unreachable: {', '.join(skipped)}" if skipped else "")
    return RedirectResponse(url=f"/cluster?msg={quote(msg)}", status_code=303)


@app.post("/api/peers/discover")
async def api_peers_discover():
    if not conn.connected:
        return RedirectResponse(url="/", status_code=303)
    # ponytail: scans ±20 ports on the connected host, which fits kubectl port-forward setups; type URLs for anything else.
    ports = [p for p in range(max(1, conn.port - 20), conn.port + 21) if p != conn.port]
    # Closed ports fail at connect; port-forwarded peers can take ~1 s to answer.
    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=0.3)) as client:
        clus, _ = await _qget(client, "/cluster")
        self_id = str(_d(clus).get("peer_id") or "")
        known = {str(p) for p in _d(_d(clus).get("peers"))} - {self_id}
        urls = [f"http://{conn.host}:{p}" for p in ports]
        resolved = await asyncio.gather(*(_resolve_peer(client, u) for u in urls))
    found = {}
    for url, pid in zip(urls, resolved):
        if pid in known and pid not in found:
            found[pid] = url
    taken = set(found.values())
    conn.peer_urls = {pid: u for pid, u in conn.peer_urls.items() if u not in taken or found.get(pid) == u}
    conn.peer_urls.update(found)
    _save_conn_state(conn.host, conn.port, conn.peer_urls)
    missing = known - set(conn.peer_urls)
    msg = f"Discovered {len(found)} peer endpoint(s) on {conn.host}:{ports[0]}–{ports[-1]}." + (
        f" Still unmapped: {', '.join(sorted(missing))}" if missing else "")
    return RedirectResponse(url=f"/cluster?msg={quote(msg)}", status_code=303)


async def _cluster_state(request: Request) -> dict:
    """Everything the cluster page and the overview show; all Qdrant calls run in parallel."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=2.0)) as client:
        (tele, e1), (clus, e2), (cltele, e3), (quotas, e4), (issues, e5), (metrics_txt, e6), \
            (healthz, _), (readyz, e8), (snaps, e9) = await asyncio.gather(
                _qget(client, "/telemetry?details_level=3"),
                _qget(client, "/cluster"),
                _qget(client, "/cluster/telemetry?details_level=3"),
                _qget(client, "/quotas", optional=True),
                _qget(client, "/issues"),
                _qget(client, "/metrics", raw=True),
                _qget(client, "/healthz", raw=True),
                _qget(client, "/readyz", raw=True),
                _qget(client, "/snapshots"),
            )
        tele = _d(tele)
        coll_ids = [c.get("id") for c in _d(tele.get("collections")).get("collections") or [] if c.get("id")]
        (slow, e10), *rest = await asyncio.gather(
            _qget(client, "/profiler/slow_requests?limit=20", optional=True),
            *(_qget(client, f"/collections/{quote(cid, safe='')}/memory") for cid in coll_ids),
            *(_qget(client, f"/collections/{quote(cid, safe='')}") for cid in coll_ids),
        )
        mem_results, info_results = rest[:len(coll_ids)], rest[len(coll_ids):]
        info_by_coll = {cid: _d(res) for cid, (res, _) in zip(coll_ids, info_results)}

        self_id = str(_d(clus).get("peer_id") or "")
        bases = {str(pid): (_qdrant_base_url() if str(pid) == self_id else conn.peer_urls.get(str(pid)))
                 for pid in _d(_d(clus).get("peers"))}
        others = {pid: b for pid, b in bases.items() if b and pid != self_id}
        peer_fetch = await asyncio.gather(*(
            asyncio.gather(_qget(client, "/cluster", base=b), _qget(client, "/telemetry?details_level=3", base=b),
                           _qget(client, "/metrics", raw=True, base=b))
            for b in others.values()
        ))
        peer_live = {self_id: {"cluster": _d(clus), "telemetry": tele, "metrics": metrics_txt or "", "error": None}}
        for pid, ((pc, pe1), (pt, pe2), (pm, pe3)) in zip(others, peer_fetch):
            got = str(_d(pc).get("peer_id") or "")
            if got and got != pid:
                # Port-forwards get reshuffled; never show another peer's data under this id.
                bases[pid] = None
                peer_live[pid] = {"cluster": {}, "telemetry": {}, "metrics": "",
                                  "error": f"{others[pid]} now serves peer {got}; remap it"}
                continue
            peer_live[pid] = {"cluster": _d(pc), "telemetry": _d(pt), "metrics": pm or "", "error": pe1 or pe2 or pe3}
        if not bases:
            bases["local"] = _qdrant_base_url()
        tele_by_peer = {pid: v["telemetry"] for pid, v in peer_live.items()}
        replica_results = await asyncio.gather(*(_replica_views(client, bases, cid, tele_by_peer) for cid in coll_ids))
    replicas_by_coll, replica_peers = {}, {}
    replica_findings = []
    rf_by_coll = {c.get("id"): _d(_d(c.get("config")).get("params")).get("replication_factor") or 1
                  for c in _d(tele.get("collections")).get("collections") or []}
    for cid, (views, rstats, rerr, lstatus) in zip(coll_ids, replica_results):
        mx = insights.replica_matrix(views, rstats, _cluster_shards(cltele, cid), lstatus)
        replicas_by_coll[cid] = mx
        replica_peers[cid] = insights.peer_summary(mx)
        pl = insights.placement(mx, list(views), rf_by_coll.get(cid, 1))
        status_findings = [w for w in insights.collection_warnings(
            {k: v for k, v in info_by_coll.get(cid, {}).items() if k in ("status", "optimizer_status")}, None, [], [])
            if w[1].startswith(("Collection status", "Optimizer error"))]
        replica_findings += [(lvl, f"{cid}: {msg}") for lvl, msg in status_findings + mx["findings"] + pl["findings"]
                             if lvl != "info" or msg.startswith("Collection status")]
    replica_findings.sort(key=lambda f: {"error": 0, "warn": 1}.get(f[0], 2))
    # Peers whose replicas no source (direct endpoint or /cluster/telemetry) reported.
    unmapped = sorted({p for mx in replicas_by_coll.values() for p in mx["unreachable"]})
    covered = {r["peer"] for mx in replicas_by_coll.values() for sh in mx["shards"] for r in sh["replicas"]
               if r["source"] == "cluster telemetry"}
    stale = {pid: v["error"] for pid, v in peer_live.items() if v["error"] and "now serves peer" in v["error"]}
    errors_extra = list(stale.values())
    mem_by_coll = {cid: _d(m) for cid, (m, _) in zip(coll_ids, mem_results)}
    errors = [e for e in (e1, e2, e3, e4, e5, e6, e8, e9, e10) if e] + errors_extra + [e for _, e in info_results if e][:3]
    version = tuple(int(x) for x in re.findall(r"\d+", str(_d(tele.get("app")).get("version") or ""))[:2]) or (0, 0)
    mem_supported = version >= (1, 18)
    mem_errors = [e for _, e in mem_results if e] if mem_supported else []
    clus, cltele, quotas = _d(clus), _d(cltele), _d(quotas)
    metrics = insights.parse_prom(metrics_txt or "")

    app_info = _d(tele.get("app"))
    system = _d(app_info.get("system"))
    ram_total = insights.kb_to_bytes(system.get("ram_size"))
    disk_total = insights.kb_to_bytes(system.get("disk_size"))

    hw = _d(_d(tele.get("hardware")).get("collection_data"))
    coll_rows = []
    mem_totals = {k: 0 for k in ("disk_bytes", "ram_bytes", "cached_bytes", "expected_cache_bytes", "warm_cached_bytes")}
    for c in _d(tele.get("collections")).get("collections") or []:
        shards = c.get("shards") or []
        local = [_d(s.get("local")) for s in shards if s.get("local")]
        vbytes = sum(l.get("vectors_size_bytes") or 0 for l in local)
        pbytes = sum(l.get("payloads_size_bytes") or 0 for l in local)
        nbyname: dict[str, int] = {}
        for l in local:
            for k, v in _d(l.get("num_vectors_by_name")).items():
                nbyname[k] = nbyname.get(k, 0) + (v or 0)
        tiers = insights.memory_tiers(_d(c.get("config")), nbyname, vbytes, pbytes)
        usage = insights.memory_usage(mem_by_coll[c["id"]], tiers) if mem_by_coll.get(c.get("id")) else None
        if usage:
            for k in mem_totals:
                mem_totals[k] += usage["total"][k]
        states = [st for s in shards for st in _d(s.get("replicate_states")).values()]
        cfg_params = _d(_d(c.get("config")).get("params"))
        coll_rows.append({
            "name": c.get("id"),
            "points": sum(l.get("num_points") or 0 for l in local),
            "shards": len(shards),
            "replication": cfg_params.get("replication_factor") or 1,
            "states": sorted(set(states)),
            "vectors_bytes": vbytes,
            "payload_bytes": pbytes,
            "tiers": tiers,
            "usage": usage["total"] if usage else None,
            "transfers": len(c.get("transfers") or []),
            "hw": _d(hw.get(c.get("id"))),
            "replicas": replicas_by_coll.get(c.get("id")),
            "coll_status": info_by_coll.get(c.get("id"), {}).get("status"),
            "info_error": next((e for (r, e), cid2 in zip(info_results, coll_ids) if cid2 == c.get("id") and e), None),
            "indexed_vectors": info_by_coll.get(c.get("id"), {}).get("indexed_vectors_count"),
            "segments_count": info_by_coll.get(c.get("id"), {}).get("segments_count"),
            "vectors_total": sum(nbyname.values()) if nbyname else None,
            "vec_names": [
                {"name": n or "(default)", "size": _d(v).get("size"), "distance": _d(v).get("distance")}
                for n, v in (({"": cfg_params.get("vectors")} if "size" in _d(cfg_params.get("vectors"))
                              else _d(cfg_params.get("vectors"))).items())
            ],
            "sparse_names": list(_d(cfg_params.get("sparse_vectors"))),
            "optimizer_error": insights.optimizer_error(info_by_coll.get(c.get("id"), {}).get("optimizer_status")),
            "optimizer_known": "optimizer_status" in info_by_coll.get(c.get("id"), {}),
            "running_opt": insights.prom_sum(
                metrics, "collection_running_optimizations", lambda lb, n=c.get("id"): lb.get("id") == n),
        })

    peers = []
    peer_details = _d(_d(cltele.get("cluster")).get("peers"))
    for pid, p in _d(clus.get("peers")).items():
        det = _d(peer_details.get(pid))
        live = peer_live.get(str(pid))
        live_raft = _d(_d(live["cluster"]).get("raft_info")) if live else {}
        dd = _d(det.get("details"))
        if not live_raft and dd.get("term") is not None:
            # /cluster/telemetry carries each peer's own raft view but not whom it sees as leader.
            live_raft = {"role": dd.get("role"), "term": dd.get("term"), "commit": dd.get("commit"),
                         "pending_operations": dd.get("num_pending_operations"), "leader": None,
                         "from_telemetry": True}
        live_metrics = insights.parse_prom(live["metrics"]) if live else {}
        peers.append({
            "id": pid,
            "uri": p.get("uri"),
            "endpoint": bases.get(str(pid)),
            "live_error": live["error"] if live else None,
            "live_raft": live_raft,
            "resident": insights.prom_sum(live_metrics, "memory_resident_bytes"),
            "ram_total": insights.kb_to_bytes(_d(_d(_d(live["telemetry"]).get("app")).get("system")).get("ram_size")) if live else None,
            "collections": insights.prom_sum(live_metrics, "collections_total"),
            "responsive": det.get("responsive"),
            "version": _d(det.get("details")).get("version")
                       or _d(_d(_d(tele.get("cluster")).get("peer_metadata")).get(pid)).get("version"),
            "role": _d(det.get("details")).get("role"),
            "is_self": str(clus.get("peer_id")) == str(pid),
            "quota": _d(_d(quotas.get("peers")).get(pid)),
            "send_failures": _d(clus.get("message_send_failures")).get(pid),
        })

    def ratio(a, b):
        a, b = insights.prom_sum(metrics, a), insights.prom_sum(metrics, b)
        return (a, b, (a / b * 100) if a is not None and b else None)

    raft = _d(clus.get("raft_info"))
    versions = {p["version"] for p in peers if p["version"]}
    live_rafts = [p["live_raft"] for p in peers if p["live_raft"]]
    leaders = {r.get("leader") for r in live_rafts if not r.get("from_telemetry")}
    terms = {r.get("term") for r in live_rafts if r.get("term") is not None}
    quota_exceeded = {k: v for k, v in _d(_d(tele.get("quota")).get("exceeded")).items() if v}
    leader_id = str(raft.get("leader", "")) if raft.get("leader") is not None else ""
    leader_ep = bases.get(leader_id) if leader_id else None
    standalone = clus.get("status") == "disabled"
    health = [
        ("leader", bool(leader_id) or standalone,
         "standalone (cluster mode disabled)" if standalone else
         (("this peer · " if leader_id == self_id else "") + f"…{leader_id[-6:]}"
          + (f" · {leader_ep.replace('http://', '')}" if leader_ep else "")
          + f" · term {raft.get('term')}") if leader_id else "no leader elected"),
        ("healthz", healthz is not None and "passed" in healthz, (healthz or "unreachable").strip()),
        ("readyz", not e8, (readyz or e8 or "").strip()),
        ("issues", not (issues or {}).get("issues"), f"{len((issues or {}).get('issues') or [])} open"),
        ("consensus", _d(clus.get("consensus_thread_status")).get("consensus_thread_status") in (None, "working"),
         _d(clus.get("consensus_thread_status")).get("consensus_thread_status") or "n/a"),
        ("recovery mode", not _d(app_info.get("features")).get("recovery_mode"),
         "on" if _d(app_info.get("features")).get("recovery_mode") else "off"),
        ("quota", not quota_exceeded, ", ".join(quota_exceeded) + " exceeded" if quota_exceeded else "ok"),
        ("versions", len(versions) <= 1, ", ".join(sorted(versions)) or "—"),
        ("raft views", len(leaders) <= 1 and len(terms) <= 1,
         (f"{len(live_rafts)}/{len(peers) or 1} checked, agree" if len(live_rafts) > 1 else "only this peer checked")
         if len(leaders) <= 1 and len(terms) <= 1
         else f"leaders {sorted(map(str, leaders))} · terms {sorted(map(str, terms))}"),
        ("replicas", not any(lvl == "error" for lvl, _ in replica_findings),
         ", ".join(x for x in (
             f"{sum(1 for lvl, _ in replica_findings if lvl == 'error')} error(s)",
             f"{sum(1 for lvl, _ in replica_findings if lvl == 'warn')} warning(s)",
             f"{len(unmapped)} peer(s) unverified" if unmapped else "",
         ) if x and not x.startswith("0 "))
         or "ok"),
    ]

    for row in coll_rows:
        tag = f"{row['name']}: "
        mine = [lvl for lvl, msg in replica_findings if msg.startswith(tag)]
        row["errors"], row["warnings"] = mine.count("error"), mine.count("warn")

    snap_list = snaps if isinstance(snaps, list) else []
    return {
        "conn": conn,
        "errors": errors,
        "health": health,
        "replica_findings": replica_findings,
        "leader_id": leader_id,
        "self_id": self_id,
        "unmapped": unmapped,
        "peer_ids": sorted(str(p) for p in _d(clus.get("peers"))) or sorted({p for mx in replicas_by_coll.values() for p in
                                                                              {r["peer"] for sh in mx["shards"] for r in sh["replicas"]}}),
        "status_meaning": insights.COLLECTION_STATUS,
        "replica_peers": replica_peers,
        "covered": covered,
        "msg": request.query_params.get("msg", ""),
        "app_info": app_info,
        "system": system,
        "ram_total": ram_total,
        "disk_total": disk_total,
        "memory": _d(tele.get("memory")),
        "quotas": quotas,
        "search_pool": _d(tele.get("search_pool")),
        "cluster": clus,
        "cluster_cfg": _d(_d(tele.get("cluster")).get("config")),
        "raft": raft,
        "peers": peers,
        "issues": (issues or {}).get("issues") or [],
        "collections": coll_rows,
        "mem_totals": mem_totals,
        "mem_available": any(mem_by_coll.values()),
        "mem_supported": mem_supported,
        "mem_errors": mem_errors[:3],
        # Qdrant < 1.19 reports the host machine's RAM/cores, not the container's cgroup limit (qdrant#9891).
        "host_limits": version < (1, 19),
        "slow_requests": _d(slow).get("requests") or [],
        "fds": ratio("process_open_fds", "process_max_fds"),
        "mmaps": ratio("process_open_mmaps", "system_max_mmaps"),
        "latency": insights.latency_table(metrics)[:30],
        "full_snapshots": snap_list,
        "full_snapshots_bytes": sum(s.get("size") or 0 for s in snap_list),
    }


@app.get("/cluster", response_class=HTMLResponse)
async def cluster_page(request: Request):
    if not conn.connected:
        return RedirectResponse(url="/")
    return _render(request, "cluster.html", await _cluster_state(request))


@app.get("/api/cluster/sample")
async def cluster_sample():
    # Report what Qdrant answers right now; connection state is owned by connect() and the page routes.
    where = f"{conn.host}:{conn.port}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=2.0)) as client:
            resp = await client.get(f"{_qdrant_base_url()}/metrics")
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"Qdrant at {where} unreachable: {type(exc).__name__}: {exc}"}, status_code=503)
    if resp.status_code >= 400:
        return JSONResponse({"error": f"Qdrant at {where} answered HTTP {resp.status_code}"}, status_code=502)
    return {"ts": time.time(), **insights.sample(insights.parse_prom(resp.text))}


def _local_segments(telemetry: dict, collection: str) -> list:
    for c in _d(_d(telemetry).get("collections")).get("collections") or []:
        if c.get("id") == collection:
            return [{**seg, "_shard": sh.get("id")} for sh in c.get("shards") or []
                    for seg in _d(sh.get("local")).get("segments") or []]
    return []


@app.get("/api/{name}/segments", response_class=HTMLResponse)
async def collection_segments(request: Request, name: str, peer: str = Query("")):
    if not conn.connected:
        return HTMLResponse("<p class='text-red-400 p-4 text-xs'>Not connected</p>")
    q = quote(name, safe="")
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=2.0)) as client:
        clus, bases = await _peer_bases(client)
        self_id = str(clus.get("peer_id") or "")
        base = bases.get(peer) if peer else _qdrant_base_url()
        if peer and not base:
            return HTMLResponse(f"<p class='text-[#fbbf24] p-4 text-xs'>▲ Peer {html.escape(peer)} has no mapped endpoint.</p>")
        # ponytail: level 6 telemetry covers every collection on that peer; fine for a handful of collections.
        (info, e1), (tele, e2), (who, _) = await asyncio.gather(
            _qget(client, f"/collections/{q}"),
            _qget(client, "/telemetry?details_level=6", base=base),
            _qget(client, "/cluster", base=base),
        )
    served = str(_d(who).get("peer_id") or "")
    if peer and served and served != peer:
        return HTMLResponse(f"<p class='text-[#fbbf24] p-4 text-xs'>▲ {html.escape(base)} now serves peer "
                            f"{html.escape(served)}, not {html.escape(peer)}; remap it on the Cluster page.</p>")
    analysis = insights.segment_analysis(_local_segments(_d(tele), name), _d(_d(info).get("config")))
    return _render(request, "partials/segments.html", {
        "seg": analysis,
        "errors": [e for e in (e1, e2) if e],
        "peer": peer or self_id,
        "peers": [p for p, b in bases.items() if b],
        "self_id": self_id,
        "collection_name": name,
        "indexing_threshold": _d(_d(_d(info).get("config")).get("optimizer_config")).get("indexing_threshold"),
    })


@app.get("/api/{name}/health", response_class=HTMLResponse)
async def collection_health(request: Request, name: str):
    if not conn.connected:
        return HTMLResponse("<p class='text-red-400 p-4 text-xs'>Not connected</p>")
    q = quote(name, safe="")
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=2.0)) as client:
        # ponytail: telemetry has no per-collection filter, so level 6 pulls every collection's segments; fine for a handful of collections.
        (info, e1), (shard_info, e2), (opts, e3), (tele, e4), (mem, e5), (metrics_txt, e6), (slow, e7) = \
            await asyncio.gather(
                _qget(client, f"/collections/{q}"),
                _qget(client, f"/collections/{q}/cluster"),
                _qget(client, f"/collections/{q}/optimizations"),
                _qget(client, "/telemetry?details_level=6"),
                _qget(client, f"/collections/{q}/memory", optional=True),
                _qget(client, "/metrics?per_collection=true", raw=True),
                # ponytail: endpoint has no collection filter; 100 covers 32-per-type capacity for the common types.
                _qget(client, "/profiler/slow_requests?limit=100", optional=True),
            )
        clus, bases = await _peer_bases(client)
        self_id = str(clus.get("peer_id") or "")
        others = {pid: b for pid, b in bases.items() if b and pid != self_id}
        other_tele = await asyncio.gather(*(_qget(client, "/telemetry?details_level=3", base=b) for b in others.values()))
        tele_by_peer = {self_id: _d(tele), **{pid: _d(t) for pid, (t, _) in zip(others, other_tele)}}
        views, remote_stats, view_errors, local_status = await _replica_views(client, bases, name, tele_by_peer)
        cltele = None
        # Covers peers with no endpoint and peers whose endpoint is down or now serves another peer.
        if any(v is None for v in views.values()):
            cltele, cterr = await _qget(client, "/cluster/telemetry?details_level=3")
            if cterr:
                view_errors.append(cterr)
        if "local" in bases and "local" not in views and len(views) == 1:
            self_id = next(iter(views))
            bases = {self_id: bases["local"]}
        info = _d(info)
        facet_fields = [
            f for f, fi in _d(info.get("payload_schema")).items()
            if fi.get("data_type") in ("keyword", "bool", "integer") and (fi.get("points") or 0) > 0
        ]

        async def facet(field: str):
            try:
                r = await client.post(
                    f"http://{conn.host}:{conn.port}/collections/{q}/facet",
                    json={"key": field, "limit": 10, "exact": False},
                )
                hits = _d(r.json().get("result")).get("hits") if r.status_code < 400 else None
                return field, hits or []
            except (httpx.HTTPError, ValueError):
                return field, []

        facets = dict(await asyncio.gather(*(facet(f) for f in facet_fields)))

    errors = [e for e in (e1, e2, e3, e4, e5, e6, e7) if e] + view_errors
    tele = _d(tele)
    replicas = insights.replica_matrix(views, remote_stats, _cluster_shards(cltele, name), local_status)
    rf = _d(_d(info.get("config")).get("params")).get("replication_factor") or 1
    layout = insights.placement(replicas, list(views), rf)
    raft_leader = str(_d(clus.get("raft_info")).get("leader") or "")
    coll_tele = next(
        (c for c in _d(tele.get("collections")).get("collections") or [] if c.get("id") == name), {}
    )
    opt_log, nbyname, vbytes, pbytes = [], {}, 0, 0
    segments = _local_segments(tele, name)
    seg_analysis = insights.segment_analysis(segments, _d(info.get("config")))
    for s in coll_tele.get("shards") or []:
        local = _d(s.get("local"))
        opt_log += _d(local.get("optimizations")).get("log") or []
        vbytes += local.get("vectors_size_bytes") or 0
        pbytes += local.get("payloads_size_bytes") or 0
        for k, v in _d(local.get("num_vectors_by_name")).items():
            nbyname[k] = nbyname.get(k, 0) + (v or 0)

    tiers = insights.memory_tiers(_d(info.get("config")), nbyname, vbytes, pbytes)
    usage = insights.memory_usage(_d(mem), tiers, _d(info.get("payload_schema"))) if _d(mem) else None
    coll_metrics = insights.filter_metrics(insights.parse_prom(metrics_txt or ""), collection=name)
    points = info.get("points_count") or 0
    pindex = [
        {
            "field": f,
            "type": fi.get("data_type"),
            "points": fi.get("points") or 0,
            "coverage": (fi.get("points") or 0) / points * 100 if points else None,
            "flags": [k for k in ("is_tenant", "is_principal", "on_disk") if _d(fi.get("params")).get(k)]
                     + (["hnsw"] if _d(fi.get("params")).get("enable_hnsw") else []),
        }
        for f, fi in _d(info.get("payload_schema")).items()
    ]
    pindex.sort(key=lambda r: r["points"])

    return _render(request, "partials/health.html", {
        "errors": errors,
        "warnings": sorted(
            replicas["findings"] + layout["findings"] + seg_analysis["findings"] + [
                w for w in insights.collection_warnings(
                    info, shard_info if isinstance(shard_info, dict) else None, segments, tiers, usage)
                # replica_matrix reports shard state with every peer's view, so drop the single-peer duplicate.
                if not w[1].startswith("Shard ")
            ],
            key=lambda w: {"error": 0, "warn": 1}.get(w[0], 2),
        ),
        "replicas": replicas,
        "layout": layout,
        "peer_bases": bases,
        "self_id": self_id,
        "raft_leader": raft_leader,
        "replication_factor": rf,
        "collection_name": name,
        "seg": seg_analysis,
        "seg_peers": [p for p, b in bases.items() if b],
        "indexing_threshold": _d(_d(info.get("config")).get("optimizer_config")).get("indexing_threshold"),
        "states_info": insights.REPLICA_STATES,
        "status_meaning": insights.COLLECTION_STATUS,
        "shard_info": _d(shard_info),
        "optimizations": _d(opts),
        "opt_log": sorted(opt_log, key=lambda o: o.get("start_at") or "", reverse=True)[:15],
        "tiers": tiers,
        "usage": usage,
        "latency": insights.latency_table(coll_metrics),
        "slow_requests": [r for r in _d(slow).get("requests") or [] if r.get("collection_name") == name][:20],
        "search_stats": insights.search_stats(segments),
        "pindex": pindex,
        "facets": facets,
        "points": points,
        "hw": _d(_d(_d(tele.get("hardware")).get("collection_data")).get(name)),
    })


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
