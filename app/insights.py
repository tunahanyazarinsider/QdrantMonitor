from __future__ import annotations

import re

_PROM_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)")
_PROM_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prom(text: str) -> dict[str, list[tuple[dict, float]]]:
    out: dict[str, list[tuple[dict, float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        labels = dict(_PROM_LABEL.findall(m.group(2) or ""))
        out.setdefault(m.group(1), []).append((labels, value))
    return out


def prom_sum(metrics: dict, name: str, pred=None) -> float | None:
    rows = metrics.get(name)
    if rows is None:
        return None
    return sum(v for labels, v in rows if pred is None or pred(labels))


def _is_failure(labels: dict) -> bool:
    status = labels.get("status", "")
    if "method" in labels:
        return status.isdigit() and int(status) >= 400
    return status not in ("", "0")


def sample(metrics: dict) -> dict:
    return {
        "resident_bytes": prom_sum(metrics, "memory_resident_bytes"),
        "allocated_bytes": prom_sum(metrics, "memory_allocated_bytes"),
        "cpu_cores_used": prom_sum(metrics, "cpu_cores_used"),
        "requests_total": (prom_sum(metrics, "rest_responses_total") or 0)
        + (prom_sum(metrics, "grpc_responses_total") or 0),
        "failures_total": (prom_sum(metrics, "rest_responses_total", _is_failure) or 0)
        + (prom_sum(metrics, "grpc_responses_total", _is_failure) or 0),
        "major_faults_total": prom_sum(metrics, "process_major_page_faults_total"),
        "minor_faults_total": prom_sum(metrics, "process_minor_page_faults_total"),
    }


def _hist_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    buckets = sorted(buckets)
    if not buckets or buckets[-1][1] <= 0:
        return None
    target = q * buckets[-1][1]
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return le
            return prev_le + (le - prev_le) * (target - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return prev_le


def latency_table(metrics: dict) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for proto in ("rest", "grpc"):
        p = f"{proto}_responses"

        def key(labels: dict) -> tuple:
            return (proto, labels.get("method", ""), labels.get("endpoint", ""), labels.get("status", ""))

        for labels, v in metrics.get(f"{p}_total", []):
            rows[key(labels)] = {
                "proto": proto,
                "method": labels.get("method", ""),
                "endpoint": labels.get("endpoint", ""),
                "status": labels.get("status", ""),
                "failed": _is_failure(labels),
                "count": int(v),
                "avg": None,
                "max": None,
                "buckets": [],
            }
        for suffix, field in (("avg_duration_seconds", "avg"), ("max_duration_seconds", "max")):
            for labels, v in metrics.get(f"{p}_{suffix}", []):
                if key(labels) in rows:
                    rows[key(labels)][field] = v
        for labels, v in metrics.get(f"{p}_duration_seconds_bucket", []):
            if key(labels) in rows:
                le = labels.get("le", "+Inf")
                rows[key(labels)]["buckets"].append((float("inf") if le == "+Inf" else float(le), v))
    result = []
    for r in rows.values():
        p95 = _hist_quantile(r.pop("buckets"), 0.95)
        r["p95"] = min(p95, r["max"]) if p95 is not None and r.get("max") is not None else p95
        result.append(r)
    result.sort(key=lambda r: r["count"], reverse=True)
    return result


def kb_to_bytes(kb: int | float | None) -> int | None:
    # Qdrant telemetry reports app.system.ram_size / disk_size in KiB.
    return int(kb * 1024) if kb is not None else None


def _quantization_kind(qcfg: dict | None) -> tuple[str, dict] | None:
    if not isinstance(qcfg, dict):
        return None
    for kind in ("scalar", "product", "binary"):
        if isinstance(qcfg.get(kind), dict):
            return kind, qcfg[kind]
    return None


def _quantized_bytes(kind: str, num_vectors: int, dim: int) -> int | None:
    if kind == "scalar":
        return num_vectors * dim
    if kind == "binary":
        return num_vectors * ((dim + 7) // 8)
    return None


def memory_tiers(config: dict, num_vectors_by_name: dict | None = None,
                 vectors_bytes: int | None = None, payload_bytes: int | None = None) -> list[dict]:
    """Effective memory tier per component, following the v1.19 memory-tiers mapping
    (explicit `memory` wins, else legacy on_disk / always_ram / on_disk_payload)."""
    params = config.get("params") or {}
    hnsw = config.get("hnsw_config") or {}
    vectors = params.get("vectors") or {}
    if "size" in vectors:
        vectors = {"": vectors}
    counts = num_vectors_by_name or {}
    rows: list[dict] = []

    for name, vcfg in vectors.items():
        if not isinstance(vcfg, dict):
            continue
        label = name or "(default)"
        if vcfg.get("memory"):
            tier, src = vcfg["memory"], "memory"
        elif vcfg.get("on_disk") is not None:
            tier, src = ("cold" if vcfg["on_disk"] else "cached"), "on_disk (legacy)"
        else:
            tier, src = "cached", "default"
        size = vectors_bytes if len(vectors) == 1 else None
        rows.append({"component": f"vectors · {label}", "key": ("vectors", name), "tier": tier, "source": src, "bytes": size})

        q = _quantization_kind(vcfg.get("quantization_config") or config.get("quantization_config"))
        if q:
            kind, qc = q
            if qc.get("memory"):
                qtier, qsrc = qc["memory"], "memory"
            elif qc.get("always_ram"):
                qtier, qsrc = "pinned", "always_ram (legacy)"
            else:
                qtier, qsrc = ("pinned" if tier == "cached" else "cold"), "inherited"
            n = counts.get(name)
            qbytes = _quantized_bytes(kind, n, vcfg.get("size") or 0) if n is not None else None
            rows.append({"component": f"quantized ({kind}) · {label}", "key": ("quantized", name), "tier": qtier,
                         "source": qsrc, "bytes": qbytes, "estimated": qbytes is not None})

    if hnsw.get("memory"):
        htier, hsrc = hnsw["memory"], "memory"
    elif hnsw.get("on_disk") is not None:
        htier, hsrc = ("cold" if hnsw["on_disk"] else "cached"), "on_disk (legacy)"
    else:
        htier, hsrc = "cached", "default"
    rows.append({"component": "hnsw index", "key": ("hnsw",), "tier": htier, "source": hsrc, "bytes": None})

    payload_cfg = params.get("payload") if isinstance(params.get("payload"), dict) else {}
    if payload_cfg.get("memory"):
        ptier, psrc = payload_cfg["memory"], "memory"
    elif params.get("on_disk_payload") is not None:
        ptier, psrc = ("cold" if params["on_disk_payload"] else "cached"), "on_disk_payload (legacy)"
    else:
        ptier, psrc = "cold", "default"
    rows.append({"component": "payload", "key": ("payload",), "tier": ptier, "source": psrc, "bytes": payload_bytes})
    return rows


def filter_metrics(metrics: dict, **labels) -> dict:
    return {
        name: [(lb, v) for lb, v in rows if all(lb.get(k) == want for k, want in labels.items())]
        for name, rows in metrics.items()
    }


_USAGE_KEYS = ("disk_bytes", "ram_bytes", "cached_bytes", "expected_cache_bytes")


def _usage(u) -> dict | None:
    if not isinstance(u, dict) or not any(k in u for k in _USAGE_KEYS):
        return None
    out = {k: u.get(k) or 0 for k in _USAGE_KEYS}
    out["warm_cached_bytes"] = min(out["cached_bytes"], out["expected_cache_bytes"])
    out["warm_pct"] = (out["warm_cached_bytes"] / out["expected_cache_bytes"] * 100
                       if out["expected_cache_bytes"] else None)
    return out


def memory_usage(mem: dict, tiers: list[dict], payload_schema: dict | None = None) -> dict:
    """Flattens GET /collections/{c}/memory (v1.18+) and joins each row with its configured tier."""
    tier_by_key = {t["key"]: t for t in tiers if "key" in t}
    schema = payload_schema or {}
    rows: list[dict] = []

    def add(group: str, name: str, part: str, usage, tier: dict | None, source: str = "") -> dict | None:
        u = _usage(usage)
        if u is None:
            return None
        row = {"group": group, "name": name, "part": part,
               "tier": tier["tier"] if tier else None,
               "source": tier["source"] if tier else source, **u}
        rows.append(row)
        return row

    for v in mem.get("vectors") or []:
        name = v.get("name", "")
        add("vectors", name or "(default)", "storage", v.get("storage"), tier_by_key.get(("vectors", name)))
        add("vectors", name or "(default)", "hnsw index", v.get("index"), tier_by_key.get(("hnsw",)))
        add("vectors", name or "(default)", "quantized", v.get("quantized"), tier_by_key.get(("quantized", name)))
    for v in mem.get("sparse_vectors") or []:
        name = v.get("name", "")
        add("sparse", name, "storage", v.get("storage"), None)
        add("sparse", name, "index", v.get("index"), None)
    add("payload", "", "storage", mem.get("payload"), tier_by_key.get(("payload",)))
    for pi in mem.get("payload_index") or []:
        field = pi.get("name", "")
        params = (schema.get(field) or {}).get("params") or {}
        if params.get("memory"):
            t = {"tier": params["memory"], "source": "memory"}
        elif params.get("on_disk"):
            t = {"tier": "cold", "source": "on_disk (legacy)"}
        else:
            t = {"tier": "pinned", "source": "default"}
        row = add("payload index", field, "", pi.get("usage"), t)
        if row is not None:
            row["indexed_points"] = (schema.get(field) or {}).get("points")
    for name, usage in (mem.get("other") or {}).items():
        add("other", name, "", usage, {"tier": "pinned", "source": "default"} if name == "id_tracker" else None)

    total = _usage(mem.get("total")) or {k: sum(r[k] for r in rows) for k in _USAGE_KEYS}
    total.setdefault("warm_cached_bytes", 0)
    # Cache held by cold parts (expected 0) must not count toward warmth of the parts that need it.
    if rows:
        total["warm_cached_bytes"] = sum(r["warm_cached_bytes"] for r in rows)
        total["expected_cache_bytes"] = sum(r["expected_cache_bytes"] for r in rows)
    total["warm_pct"] = (total["warm_cached_bytes"] / total["expected_cache_bytes"] * 100
                         if total["expected_cache_bytes"] else None)
    return {"rows": rows, "total": total}


# Semantics from qdrant lib/collection/src/shards/replica_set/replica_set_state.rs (v1.20).
REPLICA_STATES: dict[str, dict] = {
    "Active": {"readable": True, "updatable": True, "auto_recovery": False, "meaning": "Active and sound."},
    "Dead": {"readable": False, "updatable": False, "auto_recovery": True,
             "meaning": "Failed; Qdrant schedules automatic recovery from a healthy replica."},
    "Partial": {"readable": False, "updatable": True, "auto_recovery": False,
                "meaning": "Partially loaded, receiving data from another replica (shard transfer)."},
    "Initializing": {"readable": False, "updatable": True, "auto_recovery": False,
                     "meaning": "Collection is being created."},
    "Listener": {"readable": False, "updatable": True, "auto_recovery": False,
                 "meaning": "Receives writes but is not used for search (backup replica)."},
    "PartialSnapshot": {"readable": False, "updatable": False, "auto_recovery": False,
                        "meaning": "Legacy snapshot transfer in progress (1.7–1.8)."},
    "Recovery": {"readable": False, "updatable": False, "auto_recovery": False,
                 "meaning": "Being recovered by an external node; rejects normal updates."},
    "Resharding": {"readable": False, "updatable": True, "auto_recovery": False,
                   "meaning": "Receiving points while resharding up."},
    "ReshardingScaleDown": {"readable": True, "updatable": True, "auto_recovery": False,
                            "meaning": "Acts as Active until a resharding-down is committed."},
    "ActiveRead": {"readable": True, "updatable": True, "auto_recovery": False,
                   "meaning": "Active for readers, Partial for writers (transfer finishing)."},
    "ManualRecovery": {"readable": False, "updatable": False, "auto_recovery": False,
                       "meaning": "Created for manual recovery, usually a snapshot upload. Takes no updates, serves no reads "
                                  "and is never auto-recovered: finish the snapshot recovery or drop/re-replicate it."},
}


def replica_matrix(views: dict[str, dict | None], remote_stats: dict[str, list] | None = None,
                   cluster_shards: list | None = None) -> dict:
    """views: peer_id -> that peer's GET /collections/{c}/cluster result (None if not reachable).
    remote_stats: viewer peer_id -> telemetry shards[].remote[] (per-replica search/update counters).
    cluster_shards: GET /cluster/telemetry?details_level=3 collections[c].shards, which the connected node
    gathers from every peer; it fills replicas whose own peer is not reachable directly."""
    remote_stats = remote_stats or {}
    shards: dict[int, dict[str, dict]] = {}

    def slot(shard_id, peer):
        return shards.setdefault(shard_id, {}).setdefault(str(peer), {
            "peer": str(peer), "state": None, "points": None, "shard_key": None,
            "seen": {}, "ops": {}, "source": None,
        })

    for viewer, info in views.items():
        if not isinstance(info, dict):
            continue
        viewer = str(info.get("peer_id") or viewer)
        for ls in info.get("local_shards") or []:
            r = slot(ls.get("shard_id"), viewer)
            r["state"], r["points"], r["shard_key"] = ls.get("state"), ls.get("points_count"), ls.get("shard_key")
            r["source"] = "peer"
            r["seen"][viewer] = ls.get("state")
        for rs in info.get("remote_shards") or []:
            r = slot(rs.get("shard_id"), rs.get("peer_id"))
            r["seen"][viewer] = rs.get("state")
            r["shard_key"] = r["shard_key"] or rs.get("shard_key")

    for sh in cluster_shards or []:
        for rp in sh.get("replicas") or []:
            if rp.get("peer_id") is None:
                continue
            r = slot(sh.get("id"), rp.get("peer_id"))
            if r["state"] is None:
                r["state"], r["points"] = rp.get("state"), rp.get("num_points")
                r["source"] = "cluster telemetry"
                r["seen"].setdefault(r["peer"], rp.get("state"))

    for viewer, remotes in remote_stats.items():
        for rm in remotes or []:
            r = shards.get(rm.get("shard_id"), {}).get(str(rm.get("peer_id")))
            if r is None:
                continue
            up, se = rm.get("updates") or {}, rm.get("searches") or {}
            r["ops"][str(viewer)] = {
                "updates": up.get("count") or 0, "update_failures": up.get("fail_count") or 0,
                "searches": se.get("count") or 0, "search_failures": se.get("fail_count") or 0,
                "last": max(up.get("last_responded") or "", se.get("last_responded") or ""),
            }

    findings: list[tuple[str, str]] = []
    out = []
    for shard_id in sorted(shards, key=lambda x: (x is None, x)):
        replicas = sorted(shards[shard_id].values(), key=lambda r: r["peer"])
        counted = [r["points"] for r in replicas if r["points"] is not None]
        top = max(counted) if counted else None
        readable = 0
        for r in replicas:
            states = set(r["seen"].values())
            r["consensus_state"] = r["state"] or (next(iter(states)) if len(states) == 1 else None)
            r["disagree"] = len(states) > 1
            r["readable"] = all(REPLICA_STATES.get(s, {}).get("readable") for s in states) if states else None
            r["delta"] = (r["points"] - top) if top is not None and r["points"] is not None else None
            readable += bool(r["readable"])
            label = f"shard {shard_id} on peer {r['peer']}"
            if r["disagree"]:
                by = ", ".join(f"{v} → {s}" for v, s in sorted(r["seen"].items()))
                findings.append(("error", f"Peers disagree on {label}: {by}. Clients routed to different peers see a different cluster."))
            bad = sorted(s for s in states if s != "Active")
            for s in bad:
                lvl = "error" if not REPLICA_STATES.get(s, {}).get("readable") else "warn"
                findings.append((lvl, f"{label} is {s}: {REPLICA_STATES.get(s, {}).get('meaning', 'unknown state')}"))
            if top and r["points"] == 0:
                findings.append(("error", f"{label} holds 0 points while another replica holds {top:,}."))
            elif r["delta"] and top and abs(r["delta"]) / top > 0.001:
                findings.append(("warn", f"{label} differs by {r['delta']:+,} points ({r['delta'] / top:+.2%}) from the largest replica."))
            for viewer, op in r["ops"].items():
                if op["update_failures"]:
                    findings.append(("warn", f"Peer {viewer} failed {op['update_failures']} update(s) forwarded to {label} since it started (lifetime counter)."))
        if readable == 0:
            findings.append(("error", f"Shard {shard_id} has no readable replica: reads of this shard fail."))
        out.append({"shard_id": shard_id, "replicas": replicas, "readable": readable, "total": len(replicas),
                    "max_points": top})
    covered = {r["peer"] for sh in out for r in sh["replicas"] if r["state"] is not None}
    unreachable = sorted(str(p) for p, v in views.items() if v is None and str(p) not in covered)
    if unreachable:
        findings.append(("info", f"No data for peer(s) {', '.join(unreachable)}: their replica state and point counts are not verified."))
    findings.sort(key=lambda f: {"error": 0, "warn": 1}.get(f[0], 2))
    return {"shards": out, "findings": findings, "unreachable": unreachable}


def placement(matrix: dict, peers: list[str], replication_factor: int) -> dict:
    """Shard × peer layout plus balance checks; built on replica_matrix() output."""
    shards = matrix.get("shards") or []
    peers = sorted(set(peers) | {r["peer"] for sh in shards for r in sh["replicas"]})
    grid, findings = [], []
    per_peer = {p: {"replicas": 0, "points": 0, "sole_readable": 0} for p in peers}
    keyed = any(r.get("shard_key") is not None for sh in shards for r in sh["replicas"])

    for sh in shards:
        cells = {r["peer"]: r for r in sh["replicas"]}
        grid.append({"shard_id": sh["shard_id"], "shard_key": next((r["shard_key"] for r in sh["replicas"] if r.get("shard_key") is not None), None),
                     "points": sh["max_points"], "cells": cells})
        for r in sh["replicas"]:
            per_peer[r["peer"]]["replicas"] += 1
            per_peer[r["peer"]]["points"] += r["points"] or 0
        readable = [r for r in sh["replicas"] if r["readable"]]
        under = sh["total"] < replication_factor
        if len(readable) == 1 and not under:
            per_peer[readable[0]["peer"]]["sole_readable"] += 1
        if under:
            findings.append(("warn", f"Shard {sh['shard_id']} has {sh['total']} replica(s), below replication_factor={replication_factor}."))

    sizes = [g["points"] for g in grid if g["points"] is not None]
    if len(sizes) > 1 and not keyed and sum(sizes) >= 1000:
        mean = sum(sizes) / len(sizes)
        for g in grid:
            if g["points"] is not None and mean and g["points"] / mean >= 1.5:
                findings.append(("warn", f"Shard {g['shard_id']} holds {g['points']:,} points, {g['points'] / mean:.1f}× the shard average."))

    counts = [v["replicas"] for v in per_peer.values()]
    if len(per_peer) > 1 and counts and max(counts) - min(counts) > 1 and not keyed:
        heavy = max(per_peer, key=lambda p: per_peer[p]["replicas"])
        light = min(per_peer, key=lambda p: per_peer[p]["replicas"])
        findings.append(("warn", f"Replica placement is uneven: peer {heavy} holds {per_peer[heavy]['replicas']}, "
                                 f"peer {light} holds {per_peer[light]['replicas']}."))
    for p, v in per_peer.items():
        # With replication_factor=1 every shard has one copy by design; that is reported elsewhere.
        if v["sole_readable"] and replication_factor > 1:
            findings.append(("warn", f"Peer {p} holds the only readable replica of {v['sole_readable']} shard(s): "
                                     "if it goes down those shards cannot be read."))
    return {"peers": peers, "grid": grid, "per_peer": per_peer, "findings": findings, "keyed": keyed}


def _segment_infos(segments: list) -> list[dict]:
    return [s.get("info") or {} for s in segments if isinstance(s, dict)]


def search_stats(segments: list) -> list[dict]:
    agg: dict[tuple, dict] = {}
    for seg in segments:
        for idx in seg.get("vector_index_searches") or []:
            name = idx.get("index_name", "")
            for kind, st in idx.items():
                if kind == "index_name" or not isinstance(st, dict):
                    continue
                a = agg.setdefault((name, kind), {"index": name, "kind": kind, "count": 0,
                                                  "total_us": 0.0, "max_us": 0.0, "last": ""})
                a["count"] += st.get("count", 0) or 0
                a["total_us"] += st.get("total_duration_micros", 0) or 0
                a["max_us"] = max(a["max_us"], st.get("max_duration_micros", 0) or 0)
                a["last"] = max(a["last"], st.get("last_responded") or "")
    rows = []
    for a in agg.values():
        a["avg_us"] = a["total_us"] / a["count"] if a["count"] else None
        rows.append(a)
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


def collection_warnings(info: dict, shard_info: dict | None, segments: list,
                        tiers: list[dict], usage: dict | None = None) -> list[tuple[str, str]]:
    warns: list[tuple[str, str]] = []
    cfg = info.get("config") or {}
    params = cfg.get("params") or {}
    hnsw = cfg.get("hnsw_config") or {}
    opt = cfg.get("optimizer_config") or {}
    points = info.get("points_count") or 0

    status = info.get("status")
    if status and status != "green":
        warns.append(("error" if status == "red" else "warn", f"Collection status is {status}."))

    for shard in ((shard_info or {}).get("local_shards") or []) + ((shard_info or {}).get("remote_shards") or []):
        if shard.get("state") != "Active":
            warns.append(("error", f"Shard {shard.get('shard_id')} on peer {shard.get('peer_id', 'local')} is {shard.get('state')}."))
    if (shard_info or {}).get("shard_transfers"):
        warns.append(("warn", f"{len(shard_info['shard_transfers'])} shard transfer(s) in progress."))

    if (params.get("replication_factor") or 1) < 2:
        warns.append(("info", "replication_factor=1: a single copy of each shard; losing the node loses the data until restored."))

    if hnsw.get("m") == 0 and (hnsw.get("payload_m") or 0) > 0:
        warns.append(("info", "Global HNSW disabled (m=0, payload_m>0): only queries filtered by an is_tenant/enable_hnsw field use the index; other queries full-scan."))
    elif hnsw.get("m") == 0:
        warns.append(("warn", "HNSW disabled (m=0, payload_m=0): every search is a full scan."))

    index_disk = {r["name"]: r["disk_bytes"] for r in (usage or {}).get("rows", []) if r["group"] == "payload index"}
    for field, fi in (info.get("payload_schema") or {}).items():
        fpts = fi.get("points") or 0
        if points and fpts == 0:
            size = index_disk.get(field)
            cost = f" It still occupies {size / 1048576:.1f} MiB on disk." if size else ""
            warns.append(("warn", f"Payload index `{field}` covers 0 points: the field is indexed but never written (or misnamed).{cost}"))

    total = (usage or {}).get("total") or {}
    if total.get("warm_pct") is not None and total["warm_pct"] < 80:
        warns.append(("warn", f"Only {total['warm_pct']:.0f}% of the expected working set is in page cache; "
                              "reads will hit disk until it warms up or memory is added."))

    queue = (info.get("update_queue") or {}).get("length") or 0
    if queue:
        warns.append(("info", f"Update queue has {queue} pending operation(s)."))

    threshold = opt.get("deleted_threshold")
    for si in _segment_infos(segments):
        n = si.get("num_vectors") or 0
        deleted = si.get("num_deleted_vectors") or 0
        if threshold and n and deleted / n >= threshold * 0.8:
            warns.append(("warn", f"Segment {(si.get('uuid') or '')[:8]} has {deleted / n:.0%} deleted vectors (vacuum at {threshold:.0%})."))

    vec_tiers = {t["tier"] for t in tiers if t["component"].startswith("vectors")}
    q_tiers = {t["tier"] for t in tiers if t["component"].startswith("quantized")}
    if "cold" in vec_tiers and not q_tiers:
        warns.append(("info", "Vectors are cold and there is no quantization: searches read original vectors from disk."))
    if "cold" in vec_tiers and "pinned" in q_tiers:
        warns.append(("info", "Cold originals + pinned quantized vectors: search runs from RAM, rescoring reads disk."))
    return warns


if __name__ == "__main__":
    m = parse_prom(
        'rest_responses_total{method="GET",endpoint="/a",status="200"} 8\n'
        'rest_responses_total{method="GET",endpoint="/a",status="500"} 2\n'
        'grpc_responses_total{endpoint="/q",status="0"} 5\n'
        'grpc_responses_total{endpoint="/q",status="14"} 1\n'
        'rest_responses_duration_seconds_bucket{method="GET",endpoint="/a",status="200",le="0.1"} 4\n'
        'rest_responses_duration_seconds_bucket{method="GET",endpoint="/a",status="200",le="1"} 8\n'
        'rest_responses_duration_seconds_bucket{method="GET",endpoint="/a",status="200",le="+Inf"} 8\n'
        "memory_resident_bytes 100\n"
    )
    s = sample(m)
    assert s["requests_total"] == 16 and s["failures_total"] == 3, s
    assert s["resident_bytes"] == 100
    lat = latency_table(m)
    p95 = next(r for r in lat if r["status"] == "200" and r["proto"] == "rest")["p95"]
    assert abs(p95 - (0.1 + 0.9 * (7.6 - 4) / 4)) < 1e-9, p95
    capped = latency_table(parse_prom(
        'rest_responses_total{method="G",endpoint="/b",status="200"} 4\n'
        'rest_responses_max_duration_seconds{method="G",endpoint="/b",status="200"} 0.002\n'
        'rest_responses_duration_seconds_bucket{method="G",endpoint="/b",status="200",le="0.001"} 0\n'
        'rest_responses_duration_seconds_bucket{method="G",endpoint="/b",status="200",le="0.005"} 4\n'))
    assert capped[0]["p95"] == 0.002, capped
    assert kb_to_bytes(16777216) == 16 * 1024**3

    cfg = {
        "params": {"vectors": {"v": {"size": 256, "on_disk": True}}, "on_disk_payload": False},
        "hnsw_config": {"m": 0, "payload_m": 16, "on_disk": False},
        "quantization_config": {"scalar": {"type": "int8", "always_ram": True}},
    }
    tiers = {t["component"]: t for t in memory_tiers(cfg, {"v": 10}, 1000, 50)}
    assert tiers["vectors · v"]["tier"] == "cold"
    assert tiers["quantized (scalar) · v"]["tier"] == "pinned"
    assert tiers["quantized (scalar) · v"]["bytes"] == 2560
    assert tiers["hnsw index"]["tier"] == "cached"
    assert tiers["payload"]["tier"] == "cached"
    assert memory_tiers({"params": {"vectors": {"size": 4, "memory": "cold"}}})[0]["tier"] == "cold"
    assert memory_tiers({"params": {"vectors": {"size": 4}}})[-1]["tier"] == "cold"

    info = {"status": "green", "points_count": 10, "config": {**cfg, "optimizer_config": {"deleted_threshold": 0.2}},
            "payload_schema": {"brand": {"points": 0}, "cat": {"points": 10}}}
    segs = [{"info": {"uuid": "abcdefgh12", "num_vectors": 10, "num_deleted_vectors": 2}}]
    w = [msg for _, msg in collection_warnings(info, {"local_shards": [{"shard_id": 0, "state": "Dead"}]},
                                               segs, list(tiers.values()))]
    assert any("`brand`" in x for x in w) and not any("`cat`" in x for x in w)
    assert any("Dead" in x for x in w) and any("20% deleted" in x for x in w)
    assert any("m=0" in x for x in w)

    st = search_stats([{"vector_index_searches": [{"index_name": "v", "filtered_small_cardinality": {
        "count": 2, "total_duration_micros": 10, "max_duration_micros": 7}}]},
        {"vector_index_searches": [{"index_name": "v", "filtered_small_cardinality": {
            "count": 3, "total_duration_micros": 20, "max_duration_micros": 9}}]}])
    assert st[0]["count"] == 5 and st[0]["avg_us"] == 6 and st[0]["max_us"] == 9
    mem = {
        "total": {"disk_bytes": 100, "ram_bytes": 10, "cached_bytes": 30, "expected_cache_bytes": 60},
        "vectors": [{"name": "v", "storage": {"disk_bytes": 50, "ram_bytes": 0, "cached_bytes": 1, "expected_cache_bytes": 0},
                     "index": {"disk_bytes": 5, "ram_bytes": 0, "cached_bytes": 5, "expected_cache_bytes": 5},
                     "quantized": {"disk_bytes": 9, "ram_bytes": 9, "cached_bytes": 0, "expected_cache_bytes": 0}}],
        "payload": {"disk_bytes": 20, "ram_bytes": 0, "cached_bytes": 10, "expected_cache_bytes": 20},
        "payload_index": [{"name": "brand", "usage": {"disk_bytes": 2097152, "ram_bytes": 0, "cached_bytes": 0,
                                                      "expected_cache_bytes": 0}}],
        "other": {"id_tracker": {"disk_bytes": 1, "ram_bytes": 1, "cached_bytes": 1, "expected_cache_bytes": 0}},
    }
    mu = memory_usage(mem, list(tiers.values()), {"brand": {"points": 0, "params": {}}})
    by = {(r["group"], r["part"]): r for r in mu["rows"]}
    assert by[("vectors", "storage")]["tier"] == "cold" and by[("vectors", "quantized")]["tier"] == "pinned"
    assert by[("vectors", "hnsw index")]["warm_pct"] == 100.0 and by[("payload", "storage")]["warm_pct"] == 50.0
    assert by[("payload index", "")]["tier"] == "pinned" and by[("other", "")]["tier"] == "pinned"
    assert mu["total"]["warm_pct"] == 60.0, mu["total"]
    w2 = [msg for _, msg in collection_warnings(info, None, [], list(tiers.values()), mu)]
    assert any("`brand`" in x and "2.0 MiB" in x for x in w2) and any("60% of the expected" in x for x in w2)
    fm = filter_metrics(parse_prom('a{collection="x"} 1\na{collection="y"} 2\n'), collection="x")
    assert fm["a"] == [({"collection": "x"}, 1.0)]
    # Fixture: real 3-peer cluster where peer A reports itself Active with 0 points, B/C see it as ManualRecovery.
    A, B, C = "1238546195235563", "7280789038393895", "6268658232539136"
    views = {
        A: {"peer_id": int(A), "local_shards": [{"shard_id": 0, "points_count": 0, "state": "Active"}],
            "remote_shards": [{"shard_id": 0, "peer_id": int(B), "state": "Active"},
                              {"shard_id": 0, "peer_id": int(C), "state": "Active"}]},
        B: {"peer_id": int(B), "local_shards": [{"shard_id": 0, "points_count": 991806, "state": "Active"}],
            "remote_shards": [{"shard_id": 0, "peer_id": int(A), "state": "ManualRecovery"},
                              {"shard_id": 0, "peer_id": int(C), "state": "Active"}]},
        C: {"peer_id": int(C), "local_shards": [{"shard_id": 0, "points_count": 991806, "state": "Active"}],
            "remote_shards": [{"shard_id": 0, "peer_id": int(B), "state": "Active"},
                              {"shard_id": 0, "peer_id": int(A), "state": "ManualRecovery"}]},
    }
    stats = {A: [{"shard_id": 0, "peer_id": int(B), "updates": {"count": 0, "fail_count": 10}, "searches": {"count": 0}}]}
    mx = replica_matrix(views, stats)
    reps = {r["peer"]: r for r in mx["shards"][0]["replicas"]}
    assert reps[A]["disagree"] and reps[A]["readable"] is False and reps[A]["points"] == 0
    assert not reps[B]["disagree"] and reps[B]["readable"] and reps[B]["delta"] == 0
    assert mx["shards"][0]["readable"] == 2 and mx["shards"][0]["max_points"] == 991806
    msgs = [m for _, m in mx["findings"]]
    assert mx["findings"][0][0] == "error"
    assert any("disagree" in m for m in msgs) and any("ManualRecovery" in m for m in msgs)
    assert any("0 points while another replica holds 991,806" in m for m in msgs)
    assert any("failed 10 update" in m for m in msgs)
    partial = replica_matrix({A: views[A], B: None})
    assert partial["unreachable"] == [B] and partial["shards"][0]["replicas"][1]["points"] is None
    assert collection_warnings({"status": None, "points_count": None, "config": {}}, {"local_shards": None}, [], []) is not None
    assert replica_matrix({"x": {"peer_id": None, "local_shards": [{"shard_id": 0, "points_count": None, "state": "Active"}]}})["shards"][0]["replicas"][0]["peer"] == "x"
    lat404 = latency_table(parse_prom('rest_responses_total{method="GET",endpoint="/x",status="404"} 2\n'))
    assert lat404[0]["avg"] is None and lat404[0]["max"] is None and lat404[0]["failed"]
    cold_hides = memory_usage({"vectors": [{"name": "v",
        "storage": {"disk_bytes": 1, "ram_bytes": 0, "cached_bytes": 1000, "expected_cache_bytes": 0},
        "index": {"disk_bytes": 1, "ram_bytes": 0, "cached_bytes": 0, "expected_cache_bytes": 1000}}],
        "total": {"disk_bytes": 2, "ram_bytes": 0, "cached_bytes": 1000, "expected_cache_bytes": 1000}}, [])
    assert cold_hides["total"]["warm_pct"] == 0.0, cold_hides["total"]
    odd = memory_usage({"something_else": 1}, [])
    assert odd["rows"] == [] and odd["total"]["warm_cached_bytes"] == 0 and odd["total"]["warm_pct"] is None
    skip = memory_usage({"vectors": [{"name": "brand", "quantized": {"disk_bytes": 1}}],
                         "payload_index": [{"name": "brand", "usage": None}]}, [], {"brand": {"points": 7}})
    assert all("indexed_points" not in r for r in skip["rows"]), skip["rows"]
    via_ct = replica_matrix({A: views[A], B: None, C: None}, None, [{"id": 0, "replicas": [
        {"peer_id": int(A), "state": "Active", "num_points": 0},
        {"peer_id": int(B), "state": "Active", "num_points": 991806},
        {"peer_id": int(C), "state": "Active", "num_points": 991806}]}])
    vr = {r["peer"]: r for r in via_ct["shards"][0]["replicas"]}
    assert vr[B]["points"] == 991806 and vr[B]["source"] == "cluster telemetry" and vr[A]["source"] == "peer"
    assert via_ct["unreachable"] == [] and vr[A]["delta"] == -991806
    # 6 shards × RF 2 over 3 peers: shard 5 under-replicated and skewed, shard 2 has one Dead copy.
    P = ["p1", "p2", "p3"]
    def rep_(sid, peer, pts, st="Active"):
        return {"shard_id": sid, "points_count": pts, "state": st}
    views6 = {p: {"peer_id": p, "local_shards": [], "remote_shards": []} for p in P}
    layout = {0: ["p1", "p2"], 1: ["p2", "p3"], 2: ["p3", "p1"], 3: ["p1", "p2"], 4: ["p2", "p3"], 5: ["p1"]}
    for sid, holders in layout.items():
        for h in holders:
            st = "Dead" if (sid == 2 and h == "p1") else "Active"
            pts = 50000 if sid == 5 else 10000
            views6[h]["local_shards"].append(rep_(sid, h, pts, st))
            for other in P:
                if other != h:
                    views6[other]["remote_shards"].append({"shard_id": sid, "peer_id": h, "state": st})
    mx6 = replica_matrix(views6)
    pl = placement(mx6, P, 2)
    pm = [m for _, m in pl["findings"]]
    assert len(pl["grid"]) == 6 and pl["per_peer"]["p1"]["replicas"] == 4 and pl["per_peer"]["p3"]["replicas"] == 3
    assert any("Shard 5 has 1 replica" in m for m in pm)
    assert any("Shard 5 holds 50,000" in m for m in pm)
    assert any("Peer p3 holds the only readable replica of 1" in m for m in pm), pm
    assert not any("Peer p1 holds the only readable replica" in m for m in pm), pm
    kv = {"a": {"peer_id": "a", "local_shards": [{"shard_id": i, "points_count": 1, "state": "Active", "shard_key": f"k{i}"} for i in range(3)]},
          "b": {"peer_id": "b", "local_shards": []}}
    assert placement(replica_matrix(kv), ["a", "b"], 1)["findings"] == []
    ro = replica_matrix({"a": {"peer_id": "a", "local_shards": [], "remote_shards": [{"shard_id": 0, "peer_id": "b", "state": "Active"}]}, "b": None})
    assert ro["shards"][0]["replicas"][0]["source"] is None
    assert any("Dead" in m for _, m in mx6["findings"]) and mx6["shards"][2]["readable"] == 1
    single = placement(replica_matrix({"x": {"peer_id": "x", "local_shards": [{"shard_id": 0, "points_count": 5, "state": "Active"}]}}), ["x"], 1)
    assert single["findings"] == [], single["findings"]
    print("ok")
