#!/usr/bin/env python3
"""Stage A knob PLANNER: derive search-time tuning from measurable dataset properties.

Every constant traces to a design law (DESIGN_LAWS.md), an engine preset
(sbann-rs/src/main.rs SearchPreset, lines ~2514-2650), or a cited FINDINGS default
(P295/P321/P341/P343/P350/P353/P360/P362/P363/P364).  Where the laws underdetermine
a knob the plan says so (annotation contains UNDERDETERMINED / PILOT) instead of
fitting the 108 tuned_optima rows.

Modes:
  plan:       --d --n --metric --ood --graph-k --ram-gb [--kf] --target R  (or --band frontier)
  retrodict:  --retrodict  -> parses experiments/tuned_optima.md, scores the planner
              against every recovered knob, writes plan_knobs_retrodiction.md
"""
import argparse, json, math, os, re, sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIMA = os.path.join(HERE, "tuned_optima.md")
SCORECARD = os.path.join(HERE, "plan_knobs_retrodiction.md")

# ---------------------------------------------------------------- law tags (annotations)
L = {
    "band":   "engine SearchPreset::from_target (main.rs:2514): fast<=0.91, balanced<=0.96, accurate above",
    "cost1":  "cost-L1 touched-lines x residency-tier; two-number rule (lines>=4 + memory-bound => SQ4) CO-GATED by sqrt(m) precision margin -- cohere-10M d=768 has 12-line rows yet SQ4 is non-Pareto (-0.003 recall tax, P345)",
    "cost2":  "cost-L2 scatter-vs-stream: per-dataset pilot inequality rho x E_s/E_a, NOT a dimension threshold (msturing d=100 cascade wins every band; wiki d=1024 walk wins a mid band)",
    "cost5":  "cost-L5 cascade fixed floor; engine survivor_floor (main.rs:2559-2582): d-banded 10M base x sqrt(n/10M) clamped [0.5,1.0], min 128 (P364 capped floor scaling)",
    "rec1":   "recall-L1 containment: recall moved only by stage-cut containment; width knobs sized by containment pilot",
    "rec2":   "recall-L2 sqrt(m)/margin: cut bits to threshold then stop; fp16+12-row f32 correction band = unconditional final tier (P350)",
    "rec4":   "recall-L4 reachability miss(R)=m_inf+A*delta^R; delta is a per-dataset PILOT quantity, hop/probe ladders do NOT transfer (tuning-transfer taxonomy)",
    "edge":   "data-structure L1 edge-budget: k* grows with n (P335 k32@10M, P360 k64@35M); OOD co:base ratio scales with shift (P321 16:16 moderate, P343 48:16 extreme); capped by edges materialized in the graph file",
    "w363":   "P363 rerank-width law: latency W=1.6k, balanced W=2k, accurate W=4.8k (k=10 -> 16/20/48; engine cascade_k main.rs:2545); containment-pilot escalation where W=2k loses containment (WebVid kept W>=32)",
    "p364":   "P364 encoded defaults: kedge=64, loose-band T=250, W12/16 low/mid band at d=1024, M and p grow together with target recall, wiki balanced probe ladder [9,19,38,76]",
    "probes": "engine default_probe_ladder (main.rs:2623): center = reference_probes(band,d) x Kf/65536 x min(1, 10M/n) (scanned mass ~ p*n/Kf preserved above 10M); ladder [c/2,c,2c,4c], frontier tail extends to 16c (P353 measured to p=608 ~ 13c on DEEP)",
    "rung":   "rung within band by geometric miss-halving (recall-L4 saturation, delta~0.5 per work doubling -- stated approximation of measured 0.44-0.72)",
    "gamma":  "coverage-L3 calibration carve-out: gamma is the ONLY query-time cell-level lever; OOD-scoped, refit per dataset (cited: t2i 0.5 P206/P215, WebVid 0.2 P343); in-dist gamma=1 (unset)",
    "qseed":  "query-density starvation law: query-aware materialization needs shift AND train-query density >= ~1-2% (wiki 1.4% starved -> null; WebVid dense -> frontier flip); S ladder cited from P341/P343 (64/256/768)",
    "hops":   "engine graph_hops (main.rs:2529) 1/2/3 by band; -1 at d>=1024 (wiki tuned h1 low/mid, h2 tail; scattered-eval cost boundary P361). Escalation past preset (WebVid h4-10, msturing h3-8) needs the delta pilot -- UNDERDETERMINED from descriptor",
    "gm":     "engine graph_m (main.rs:2537) 16/24/48 by band; P364 says M grows with p along the tail -- exact schedule UNDERDETERMINED (no law constant)",
    "i8k":    "SQ4->int8 promotion width; cited per-regime defaults only: d>=1024 256/512 (P360/P364), 512<=d<1024 192/384 (P341); scaling law UNDERDETERMINED beyond these anchors",
    "res":    "cost-L1 residency auto-set: RESIDENT_I8 when int8 footprint <= RAM/2 (P346 +27.6% DEEP, P341 wiki 35.8GB anon). Known gap: pays only under memory pressure at small footprints (msturing null P345; memory-pressure regime law is an acknowledged GAP #2)",
    "arch":   "role-reallocation L5 / cost-L2: round_walk candidate at loose band + rows <=2 cache lines (DEEP P353/P356); PILOT-GATED -- msturing d=100 is the standing counterexample (cascade wins every band)",
    "goff":   "graph off: in-dist n<=1M coverage is probe-sufficient, extra materialization dead weight (coverage-L3 P303/P335-kills; cohere-1M tuned graph-free P304). Graph-free p/TFLOOR are UNDERDETERMINED (single anchor: P304 ladder, P295 t=31.25p)",
}

# ---------------------------------------------------------------- engine preset mirror
BANDS = ("fast", "balanced", "accurate")
BAND_LO = {"fast": 0.80, "balanced": 0.91, "accurate": 0.96}

def band_of(target):
    return "fast" if target <= 0.91 else ("balanced" if target <= 0.96 else "accurate")

def lines_per_row(d):  # int8 row bytes = d
    return math.ceil(d / 64)

def eng_survivor_floor(band, n, d):  # main.rs:2559-2582
    bi = 0 if d <= 128 else 1 if d <= 256 else 2 if d <= 512 else 3
    base = {"fast": [250, 500, 600, 250], "balanced": [450, 1000, 1200, 500],
            "accurate": [1800, 3000, 2500, 1000]}[band][bi]
    scale = min(1.0, max(0.5, math.sqrt(n / 1e7)))
    return max(128, round(base * scale))

def eng_reference_probes(band, d):  # main.rs:2586-2604
    tab = {"fast": [8, 32, 32, 32], "balanced": [8, 40, 64, 64], "accurate": [32, 128, 256, 160]}
    bi = 0 if d <= 128 else 1 if d <= 256 else 2 if d <= 512 else 3
    return tab[band][bi]

def probe_ladder(band, d, kf, n):
    ref = eng_reference_probes(band, d)
    center = max(1, min(kf, math.ceil(ref * kf / 65536.0 * min(1.0, 1e7 / n))))
    nrungs = 6 if band == "accurate" else 4     # frontier tail extension (P353)
    lad, seen = [], set()
    for r in range(nrungs):  # engine ladder [c/2, c, 2c, 4c] (main.rs:2637), e.g. wiki balanced [9,19,38,76]
        v = min(kf, max(1, center // 2 if r == 0 else center * (2 ** (r - 1))))
        if v not in seen:
            lad.append(v); seen.add(v)
    return lad

def pick_rung(band, target, ladder):
    miss, miss_lo = max(1e-4, 1.0 - target), 1.0 - BAND_LO[band]
    r = int(round(math.log2(miss_lo / miss))) if miss < miss_lo else 0
    return ladder[max(0, min(len(ladder) - 1, r))]

# ---------------------------------------------------------------- the planner
def plan(ds, target):
    """ds: dict(d, n, metric, ood in {'in','moderate','extreme'}, qdensity, graph_k, ram_gb, kf).
    Returns OrderedDict knob -> {'value','law'}."""
    d, n, kf = ds["d"], ds["n"], ds.get("kf") or 1 << round(math.log2(max(2, ds["n"] / 150)))
    band, lines = band_of(target), lines_per_row(ds["d"])
    ood, qd = ds.get("ood", "in"), ds.get("qdensity") or 0.0
    k = OrderedDict()
    def put(name, value, law): k[name] = {"value": value, "law": law}

    put("band", band, L["band"])
    # -- architecture (walk vs cascade): pilot-gated screen only
    walk = (lines <= 2 and band == "fast" and ood == "in")
    put("mechanism", "round_walk(PILOT rho*E_s/E_a)" if walk else "cascade", L["arch"])
    # -- storage tier flags (cost-L1 auto-set + precision co-gate)
    sq4 = lines >= 8   # clear regime; 4-7 lines = load-dependent gray zone (t2i d=200 pays ~0 unpressured, P345)
    put("sq4_nav", sq4, L["cost1"] + " -- ON at lines/row>=8 only; verdict still PILOT-gated (containment >= ~0.995, GATE1)")
    put("nav_tier", "sq4-sidecar" if sq4 else "int8", "nav tier by lines/row; 'lowrank' has NO law anchor -- never emitted")
    put("resident_i8", n * d <= ds.get("ram_gb", 256) * 0.5e9, L["res"])
    put("rerank_f16", True, L["rec2"] + " (F16 shortlist + F16_CORR_BAND=12 exact-f32 rows)")
    put("float_rerank", True, "exact final tier (referee-grade scoring); FLOAT_RERANK=1 on every tuned row")
    # -- graph
    graph_on = ds.get("graph_k", 0) > 0 and not (n <= 1_000_000 and ood == "in")
    put("graph", graph_on, L["goff"] if not graph_on else "adjacency carries coverage at scale/shift (coverage-L3, edge-budget L1)")
    if graph_on:
        base_k = 16 if n < 5_000_000 else 32 if n < 20_000_000 else 64
        want = base_k * (4 if ood == "extreme" and qd >= 0.02 else 1)  # extreme: 1 base + 3 co (P343 48:16)
        put("kedge", min(ds["graph_k"], want), L["edge"])
        hops = {"fast": 1, "balanced": 2, "accurate": 3}[band] - (1 if d >= 1024 else 0)
        put("hops", max(1, hops), L["hops"])
        put("graph_m", {"fast": 16, "balanced": 24, "accurate": 48}[band], L["gm"])
        put("bestfirst", True, "P256 expansion ordering: strict Pareto at matched budget")
    # -- widths
    w = {"fast": 16, "balanced": 20, "accurate": 48}[band]
    if d >= 1024 and band != "accurate":
        w = 16  # P364 wiki grid kept W12/16 below the 2k default
    put("cascade_k", w, L["w363"] + ("; d>=1024 low/mid band W12/16 per " + L["p364"] if w == 16 and d >= 1024 else ""))
    # -- probes + floor
    lad = probe_ladder(band, d, kf, n)
    put("p_ladder", lad, L["probes"])
    put("p", pick_rung(band, target, lad), L["rung"] + ("; UNDERDETERMINED graph-free (P304 single anchor)" if not graph_on else ""))
    put("tfloor", eng_survivor_floor(band, n, d), L["cost5"] + ("; UNDERDETERMINED graph-free (P295 t=31.25p single anchor)" if not graph_on else ""))
    if sq4:
        base, hi = (256, 512) if d >= 1024 else (192, 384)
        put("sq4_int8k", hi if band == "accurate" else base, L["i8k"])
    # -- OOD calibration + query-aligned seeds
    put("gamma", 0.5 if ood != "in" else None, L["gamma"])
    qseed = ood == "extreme" and qd >= 0.02
    put("qseed", qseed, L["qseed"])
    if qseed:
        put("qseed_S", {"fast": 64, "balanced": 256, "accurate": 768}[band], L["qseed"])
    put("kf_hint", 1 << round(math.log2(max(2, n / 150))),
        "info-only: coarse-cell knee ~150 pts/leaf with graph on (role-reallocation L5, P325); scale-gated (wash at 100M, P240) -- retrodiction uses the actual built Kf")
    return k

def plan_frontier(ds):
    return [(t, plan(ds, t)) for t in (0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 0.995)]

# ---------------------------------------------------------------- tuned_optima.md parsing
DATASETS = {
    "wiki35m":  dict(name="wiki-35M", d=1024, n=34_999_000, metric="ip", ood="in", qdensity=0.014, graph_k=64, kf=65536, ram_gb=256),
    "deep10m":  dict(name="DEEP-10M", d=96, n=10_000_000, metric="l2", ood="in", qdensity=0.0, graph_k=32, kf=65536, ram_gb=256),
    "webvid":   dict(name="WebVid-2.5M", d=512, n=2_500_000, metric="ip", ood="extreme", qdensity=0.10, graph_k=64, kf=4096, ram_gb=256),
    "mst30m":   dict(name="msturing30m", d=100, n=29_998_994, metric="l2", ood="in", qdensity=0.0, graph_k=32, kf=131072, ram_gb=256),
    "cohere1m": dict(name="cohere-1M", d=768, n=1_000_000, metric="ip", ood="in", qdensity=0.0, graph_k=16, kf=16384, ram_gb=256),
    "cohere10m":dict(name="cohere-10M", d=768, n=10_000_000, metric="ip", ood="in", qdensity=0.0, graph_k=16, kf=65536, ram_gb=256),
    "t2i1m":    dict(name="t2i-1M", d=200, n=1_000_000, metric="ip", ood="moderate", qdensity=0.05, graph_k=32, kf=16384, ram_gb=256),
    "t2i10m":   dict(name="t2i-10M", d=200, n=10_000_000, metric="ip", ood="moderate", qdensity=0.05, graph_k=32, kf=65536, ram_gb=256),
    "t2i100m":  dict(name="t2i-100M", d=200, n=100_000_000, metric="ip", ood="moderate", qdensity=0.05, graph_k=16, kf=524288, ram_gb=256),
}
# dataset-common flags transcribed from the per-dataset **Notes** blocks of tuned_optima.md
COMMON = {
    "wiki35m":  dict(sq4_nav=True,  resident=True,  f16=True,  gamma=False, qseed=False),
    "deep10m":  dict(sq4_nav=False, resident=None,  f16=None,  gamma=False, qseed=False),  # per-row
    "webvid":   dict(sq4_nav=True,  resident=True,  f16=False, gamma=True,  qseed=True),
    "mst30m":   dict(sq4_nav=False, resident=False, f16=False, gamma=False, qseed=False),
    "cohere1m": dict(sq4_nav=False, resident=False, f16=False, gamma=False, qseed=False),
    "cohere10m":dict(sq4_nav=False, resident=False, f16=False, gamma=False, qseed=False),
    "t2i1m":    dict(sq4_nav=False, resident=False, f16=False, gamma=True,  qseed=False),
    "t2i10m":   dict(sq4_nav=False, resident=False, f16=False, gamma=True,  qseed=False),
    "t2i100m":  dict(sq4_nav=False, resident=False, f16=False, gamma=True,  qseed=False),
}

UNK_KW = ["unknown", "inferred", "not preserved", "not printed", "not archived",
          "not in log", "never quoted", "unrecoverable", "not quoted"]
DROP_TOKENS = {  # knob -> tokens that, just before an UNKNOWN/INFERRED marker, invalidate the field
    "p": ["p=", "plist", " p "], "tfloor": ["tfloor"], "cascade_k": ["cascade_k", "kk", "w~", "w="],
    "hops": ["hops"], "graph_m": ["m=", "/m/", "m/t"], "kedge": ["kedge"], "sq4_int8k": ["sq4_int8k", "int8k"],
    "sq4_nav": ["sq4_nav"],
}

def first_int(text, pats):
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

def parse_row(dskey, mech_cell, knob_cell, source_cell, prev):
    text = mech_cell + " | " + knob_cell
    low = text.lower()
    walk = "walk" in mech_cell
    inherit = any(s in low for s in ("common env as above", "otherwise identical", "same fp16 tail config",
                                     "same config", "identical to the", "same stack", "rest as", "common env"))
    a = {"mechanism": "walk" if walk else "cascade"}
    a["graph"] = False if "graph=off" in low else True
    a["p"] = first_int(text, [r"\bp\s*=\s*(\d+)", r"SBANN_PLIST=(\d+)"])
    a["tfloor"] = first_int(text, [r"(?:SBANN_)?TFLOOR\s*=\s*(\d+)"])
    a["cascade_k"] = first_int(text, [r"KLIST-W\s*=\s*(\d+)", r"W\s*\((?:SBANN_)?KLIST\)\s*=\s*(\d+)",
                                      r"(?:SBANN_)?CASCADE_K\)?\s*=\s*(\d+)", r"\bW\s*=\s*(\d+)"])
    a["graph_m"] = first_int(text, [r"(?:SBANN_)?GRAPH_M\s*=\s*(\d+)", r"\bbeam M\s*=\s*(\d+)", r"\bM\s*=\s*(\d+)"])
    a["kedge"] = first_int(text, [r"(?:SBANN_GRAPH_|GRAPH_)?KEDGE\s*=\s*(\d+)",
                                  r"(?:graph|co|hyb|nd)[-_a-z0-9]*?_k(\d+)\.u32", r"\bnd-k(\d+)\b", r"\bk(\d+) hybrid"])
    a["hops"] = first_int(text, [r"(?:GRAPH_)?HOPS\s*=\s*(\d+)", r"\bhops\s*=\s*(\d+)", r"\bh\s*=\s*(\d+)", r"\bh(\d+)\b"])
    a["sq4_int8k"] = first_int(text, [r"SQ4_INT8K\s*=\s*(\d+)"])
    a["qseed_S"] = first_int(text, [r"_S(\d+)\.u32", r"_S(\d+)\b", r"\bS\s*=\s*(\d+)", r"\bS(\d+)\b"])
    com = COMMON[dskey]
    a["sq4_nav"] = com["sq4_nav"] or ("sq4_nav=1" in low)
    f16 = ("rerank_f16" in low or "fp16" in mech_cell.lower()) and not any(s in low for s in ("no fp16", "no rerank_f16", "pre-fp16"))
    a["rerank_f16"] = com["f16"] if com["f16"] is not None else f16
    a["resident_i8"] = com["resident"] if com["resident"] is not None else ("resident" in low)
    # 'qseed_' matches artifact names (qseed_T16_S64.u32) but not prose negations ("no SQ4/RESIDENT/QSEED")
    a["gamma_on"], a["qseed_on"] = com["gamma"] or "gamma" in low, com["qseed"] or "qseed_" in low or "seed_ids" in low
    if inherit and prev:  # rows saying "common env as above" / "same ... config" inherit stack flags
        for f in ("rerank_f16", "resident_i8", "sq4_nav", "gamma_on", "qseed_on"):
            a[f] = prev.get(f, a[f])
        for f in ("tfloor", "cascade_k", "graph_m", "kedge", "hops", "sq4_int8k"):
            if a[f] is None:
                a[f] = prev.get(f)
    if walk:  # walk rows: cascade-shaped knobs have different semantics (R/B/W grid, P356) -- not scored
        for f in ("p", "tfloor", "cascade_k", "graph_m", "hops"):
            a[f] = None
    # UNKNOWN / INFERRED field invalidation (task rule: skip flagged fields in scoring).
    # The flagged knob names sit immediately before the marker in tuned_optima.md
    # ("p=2 W~48 INFERRED", "PLIST/hops/M/TFLOOR = UNKNOWN", "tfloor NOT preserved"),
    # so a tight backward window drops exactly the flagged fields.
    for kw in UNK_KW:
        for m in re.finditer(re.escape(kw), low):
            win = low[max(0, m.start() - 24):m.start()]
            for knob, toks in DROP_TOKENS.items():
                if any(t in win for t in toks):
                    a[knob] = None
    a["kf"] = first_int(knob_cell, [r"[Kk]f\s*=?\s*(\d+)"])
    return a

def parse_optima(path):
    rows, dskey, sub = [], None, None
    sec_map = [("wiki-35M", "wiki35m"), ("DEEP-10M", "deep10m"), ("WebVid", "webvid"),
               ("msturing", "mst30m"), ("cohere", "cohere"), ("text2image", "t2i")]
    sub_map = [("t2i-1M", "t2i1m"), ("t2i-10M", "t2i10m"), ("t2i-100M", "t2i100m")]
    prev = None
    with open(path) as f:
        for line in f:
            if line.startswith("## "):
                dskey = next((k for s, k in sec_map if s in line), None); sub = None; prev = None
            elif line.startswith("### "):
                sub = next((k for s, k in sub_map if s in line), None); prev = None
            elif line.startswith("| 0.") and dskey:
                cells = [c.replace("\x00", "|").strip() for c in line.replace("\\|", "\x00").split("|")]
                if len(cells) < 6:
                    continue
                recall, qps, mech, knobs, source = float(cells[1]), cells[2], cells[3], cells[4], cells[5]
                key = sub if dskey == "t2i" else dskey
                if dskey == "cohere":
                    key = "cohere1m" if "cohere-1M" in source else "cohere10m"
                actual = parse_row(key, mech, knobs, source, prev)
                prev = dict(actual)
                rows.append(dict(ds=key, recall=recall, actual=actual))
    return rows

# ---------------------------------------------------------------- scoring
SCORED = ["mechanism", "graph", "p", "tfloor", "cascade_k", "graph_m", "kedge", "hops",
          "sq4_nav", "sq4_int8k", "resident_i8", "rerank_f16", "gamma_on", "qseed_on", "qseed_S"]
PRED_KEY = {"gamma_on": "gamma", "qseed_on": "qseed"}

def grade(knob, pred, act):
    if pred is None or act is None:
        return None
    if knob == "mechanism":
        pw = "walk" in str(pred)
        return ("exact" if pw == (act == "walk") else "miss", "walk" if pw else "cascade")
    if isinstance(act, bool) or isinstance(pred, bool):
        return ("exact" if bool(pred) == bool(act) else "miss", "high" if pred and not act else "low")
    if knob == "hops":
        dv = abs(pred - act)
        return ("exact" if dv == 0 else "step" if dv == 1 else "miss", "high" if pred > act else "low")
    pred, act = float(pred), float(act)
    if pred <= 0 or act <= 0:
        return None
    r = max(pred, act) / min(pred, act)
    return ("exact" if r <= 1.25 else "step" if r <= 2.5 else "miss", "high" if pred > act else "low")

DIAGNOSES = [  # (knob, dataset-regex, direction-or-None, diagnosis)
    ("rerank_f16", ".*", "high",
     "PLANNER-vs-STALE-TUNING: recall-L2's simplification (P350) makes fp16+12-row f32 correction the unconditional final tier; "
     "it was measured only on DEEP (+14.5-35.4% QPS at exact recall) and adopted on wiki. All other frontier rows predate P350. "
     "Verdict: the historical tuning is (per the law) leaving QPS on the table off wiki/DEEP -- but the law's universality is UNTESTED there. "
     "Action: apply-and-measure (cheap), or scope the law to d-regimes where the rerank tier is line-bound."),
    ("resident_i8", "mst30m|cohere.*|t2i.*", "high",
     "LAW GAP (acknowledged, DESIGN_LAWS 'Gaps' #2): footprint<=RAM/2 predicts RESIDENT everywhere, but the lever only measurably pays "
     "under memory pressure or at large anon footprints (wiki 35.8GB, P341; DEEP +27.6%, P346) and was null on msturing (P345). "
     "The descriptor lacks the pressure term; history omitted a (mostly) null lever. Neither side refuted: default-ON is harmless, "
     "but the planner cannot discriminate DEEP(+27.6%) from msturing(null) at similar d/footprint."),
    ("sq4_nav", "cohere.*", "high",
     "THE CITED CO-GATE EXCEPTION (P345): cohere d=768 = 12-line rows, lines-rule says SQ4, yet SQ4 is non-Pareto (-0.003 recall tax). "
     "cost-L1's precision co-gate (sqrt(m)-margin containment pilot) is a MEASURED quantity the descriptor cannot supply. "
     "The planner correctly emits SQ4 as PILOT-GATED candidate; without the pilot it mispredicts exactly this dataset. Law fine, descriptor insufficient."),
    ("p", "cohere1m", None,
     "UNDERDETERMINED (graph-free): the engine ladder presumes the hybrid stack; with graph=off probes carry all coverage and the tuned "
     "ladder sits ~4-6x higher (P304 p96..p512). Only one graph-free dataset exists -- no law constant; the containment pilot (recall-L1) is the prescribed bridge."),
    ("tfloor", "cohere1m", None,
     "UNDERDETERMINED (graph-free) + missing p-coupling: cohere's cited t=31.25p law (P295) sets tfloor from p, not from the d-banded floor. "
     "The engine floor has no p term; single-dataset anchor, not generalized."),
    ("tfloor", "cohere10m", "low",
     "Same t=31.25p coupling (P295): tuned floor grows with p inside the accurate band (2000..4000) while the engine floor is flat per band (1000). "
     "Band-Level Adaptivity law says stages scale JOINTLY -- the flat per-band floor is the missing term."),
    ("tfloor", "deep10m", "high",
     "ENGINE-vs-TUNED INCONSISTENCY surfaced: the engine's accurate d<=128 floor (1800, main.rs:2572) was calibrated on msturing's accurate "
     "band (tuned t=1800 at p64); DEEP's tuned accurate band is flat at 450 for p32..608 (P348/P353) -- the floor knob is containment-inert there "
     "(TINY-SCAN class: the scan seeds the beam, width does not bind). FINDING: one d-banded constant conflates two datasets whose binding stage "
     "differs; recall-L1's containment probe, not d, must set the floor. The planner deliberately mirrors the engine rather than patching the constant."),
    ("tfloor", "t2i100m", "low",
     "SCALE-GATE: the capped sqrt(n/10M) floor scaling (P364) assumes probes shrink ~1/n above 10M so scanned mass stays constant; on t2i-100M "
     "the 3-level router (hierk3) kept tuned p at 40-104 (not ~10x smaller), so scanned mass and the tuned floor (8000) grew. "
     "Extrapolation to 100M was never claimed (P240 wash; role-reallocation L5 scale gate). Pilot territory."),
    ("tfloor", "wiki35m|mst30m|webvid|t2i10m", None,
     "SYSTEMATIC: within-band (esp. accurate) tuned tfloor grows with p (wiki ~45p at tail, mst ~20p, WebVid ladder 300..2000; cohere 31.25p) "
     "while the engine floor is flat per band -- yet DEEP is genuinely flat (450 for p32..608). The floor needs a dataset-coefficient p-coupling term "
     "(c_t from a containment pilot); a global constant would be an overfit. FINDING: law needs a term, engine preset is the miss on 3 of 5 graph datasets."),
    ("hops", "mst30m|webvid|t2i.*", "low",
     "UNDERDETERMINED: reachability (recall-L4) solves hops from measured delta; ladders do not transfer (tuning-transfer taxonomy). "
     "Tuned deep-hop regimes (mst h3-8 at 2-line rows, WebVid h4-10 coverage-starved OOD, t2i tail h4-6) all exceed the engine 1/2/3 preset. "
     "DEEP/cohere/wiki match the preset. The delta pilot, not the descriptor, must set hops beyond the band default."),
    ("cascade_k", "webvid", "low",
     "CONTAINMENT-PILOT ESCALATION branch of the P363 width law: on WebVid W=20 measurably loses containment (.0032/.0062, P363 verdict 'keep W32/48'), "
     "and the tail runs W128-256. The law names the escalation but its trigger is a measured pilot; descriptor-only prediction lands at the un-escalated 2k default."),
    ("cascade_k", "mst30m", None,
     "Scoring against an INERT knob: P363 proved W16==W256 recall-identical on msturing (genuine knob-not-binding physics, Referee-First law). "
     "The plotted rows carry arbitrary W arms (16/32/128); disagreement here is noise, not error."),
    ("cascade_k", "cohere1m|cohere10m", "high",
     "CONSERVATIVE-HIGH against a near-inert knob: the accurate 4.8k width exists to 'leave room for the WebVid-style binding cases' "
     "(engine comment, main.rs:2545-2554); cohere's margins are wide (P363 W16 arms reproduce the P304/P299 recalls exactly), so the tuned W=16 "
     "loses no containment and the pilot would settle at ~1.6k. Cost of the planner's 48: a slightly wider exact-rerank tier, no recall change. "
     "Law fine; the escalation trigger (measured containment) is absent from the descriptor in both directions."),
    ("p", "mst30m", "low",
     "The d<=128 reference-probe anchors (8/8/32, main.rs:2586) are DEEP-calibrated; msturing (clustered distribution, Kf=131072) needs "
     "4-10x more scanned mass at matched recall in the mid/high band. delta and probe ladders do NOT transfer across datasets "
     "(tuning-transfer taxonomy); the scanned-mass Kf/n scaling cannot absorb a distribution-shape coverage deficit. Pilot required."),
    ("p", "t2i1m|t2i10m", "low",
     "The ladder usually CONTAINS the tuned value (t2i-1M balanced tops out at the tuned p40 rung; t2i-10M accurate hits 128/256/512 exactly) "
     "-- the misses are the rung-position map: miss-halving-per-doubling (delta=0.5) under-places OOD datasets whose miss floor m_inf is high, "
     "so late rungs buy less than 2x (coverage saturation, P253/P255 delta up to 0.72). Rung placement needs the measured saturation delta, not a constant."),
    ("p", "t2i100m", "low",
     "Scanned-mass preservation (p ~ Kf/n) under-predicts t2i-100M ~2-4x: the 3-level router (hierk3) changes cells-per-probe geometry, and "
     "the mass law was calibrated at 10M/35M single-level. Scale extrapolation was never claimed (P240). Pilot required."),
    ("cascade_k", "deep10m|wiki35m", None,
     "Within-one-step of the 4.8k accurate default: tuned tails used 32 and 64 around the law's 48 (DEEP kk64 tail, wiki kk32/64). "
     "The law's ratio form (W ~ c*k) holds; the +/-1-step scatter is the containment margin the pilot would resolve."),
    ("graph_m", ".*", None,
     "P364: 'M and p grow together with target recall' -- the laws state the direction but no constant; the engine preset (16/24/48) is flat per band. "
     "Tuned tails escalate M to 64-128 (WebVid/mst/t2i/wiki). UNDERDETERMINED: needs the joint-scaling schedule (P362 H=E+(R-1)B pilot), not a fit."),
    ("p", "webvid", "low",
     "The miss-halving rung map assumes ~2x work per miss-halving (delta~0.5); WebVid is uniformly-hard OOD where probes buy little until seeds/graph "
     "saturate (routed-coverage regime, data-structure L3) -- tuned tail p rises 16..128 at nearly flat recall. delta must come from the 10-min coverage pilot."),
    ("mechanism", "mst30m", None,
     "THE STANDING COUNTEREXAMPLE (cost-L2): msturing d=100 sits in the walk byte-regime yet the cascade wins every band vs bug-fixed Roar (P339/P341). "
     "Architecture is a pilot inequality (rho x E_s/E_a), not a dimension rule -- the planner emits 'walk(PILOT)' and the pilot would refuse it here."),
    ("kedge", "webvid", "high",
     "Edge-budget currency fungibility: the k32-graph escalation arms traded edge width for hops+M at matched coverage (data-structure L1: "
     "adjacency is a budget). Planner predicts the k64 hybrid (which IS the loose-band champion); mid-band rows ran the k32 arm. Within the law."),
    ("kedge", "t2i.*", None,
     "Curve stitches k16 (early campaigns) and k32/hybrid rows; edge-budget law says k*=32 at 10M (P335) -- later rows agree. "
     "Early k16 rows are pre-P320 tuning, superseded in-law."),
]

def diagnose(knob, dskey, direction):
    for k, pat, dr, txt in DIAGNOSES:
        if k == knob and re.fullmatch(pat, dskey) and (dr is None or dr == direction):
            return txt
    return "UNDIAGNOSED -- needs investigation (no law statement covers this pattern; do not invent a constant)."

def retrodict(optima_path, out_path):
    rows = parse_optima(optima_path)
    per_knob = defaultdict(lambda: defaultdict(int))
    per_ds = defaultdict(lambda: defaultdict(int))
    misses = defaultdict(list)   # (ds, knob, dir) -> [(recall, pred, act)]
    details = []
    for row in rows:
        ds = dict(DATASETS[row["ds"]])
        if row["actual"].get("kf"):
            ds["kf"] = row["actual"]["kf"]
        pred = plan(ds, row["recall"])
        for knob in SCORED:
            pv = pred.get(PRED_KEY.get(knob, knob))
            pv = pv["value"] if isinstance(pv, dict) else pv
            g = grade(knob, pv, row["actual"].get(knob))
            if g is None:
                per_knob[knob]["skipped"] += 1
                continue
            verdict, direction = g
            per_knob[knob][verdict] += 1
            per_ds[row["ds"]][verdict] += 1
            if verdict == "miss":
                misses[(row["ds"], knob, direction)].append((row["recall"], pv, row["actual"].get(knob)))
            details.append((row["ds"], row["recall"], knob, pv, row["actual"].get(knob), verdict))
    systematic = {k: v for k, v in misses.items() if len(v) >= 3}
    write_scorecard(out_path, rows, per_knob, per_ds, systematic, misses)
    return rows, per_knob, per_ds, systematic

def pct(a, b):
    return "%.0f%%" % (100.0 * a / b) if b else "-"

def write_scorecard(path, rows, per_knob, per_ds, systematic, misses):
    o = []
    o.append("# Knob-planner retrodiction scorecard (Stage A)\n")
    o.append("Generated by `experiments/plan_knobs.py --retrodict` against `experiments/tuned_optima.md` "
             "(%d frontier rows). Grading: **exact** = ratio <= 1.25 (hops: equal), **step** = within one "
             "ladder step (ratio <= 2.5 / hops +-1), **miss** otherwise. UNKNOWN/INFERRED-flagged fields and "
             "walk-row cascade knobs are skipped. Honesty rule: every planner constant traces to a design law, "
             "the engine SearchPreset (main.rs:2514-2650), or a cited FINDINGS default -- nothing was fit to these rows.\n" % len(rows))
    o.append("\n## Per-knob agreement\n")
    o.append("| knob | scored | exact | step | miss | exact+step |")
    o.append("|---|---|---|---|---|---|")
    for knob in SCORED:
        c = per_knob[knob]
        n = c["exact"] + c["step"] + c["miss"]
        o.append("| %s | %d | %d | %d | %d | %s |" % (knob, n, c["exact"], c["step"], c["miss"], pct(c["exact"] + c["step"], n)))
    tot = defaultdict(int)
    for c in per_knob.values():
        for k, v in c.items():
            tot[k] += v
    n = tot["exact"] + tot["step"] + tot["miss"]
    o.append("| **all** | %d | %d | %d | %d | %s |" % (n, tot["exact"], tot["step"], tot["miss"], pct(tot["exact"] + tot["step"], n)))
    o.append("\n## Per-dataset agreement\n")
    o.append("| dataset | scored | exact | step | miss | exact+step |")
    o.append("|---|---|---|---|---|---|")
    for dskey in DATASETS:
        c = per_ds.get(dskey)
        if not c:
            continue
        n = c["exact"] + c["step"] + c["miss"]
        o.append("| %s | %d | %d | %d | %d | %s |" % (DATASETS[dskey]["name"], n, c["exact"], c["step"], c["miss"], pct(c["exact"] + c["step"], n)))
    o.append("\n## Systematic misses (same knob, same direction, >=3 rows) with diagnosis\n")
    if not systematic:
        o.append("(none)\n")
    for (dskey, knob, direction), items in sorted(systematic.items(), key=lambda kv: -len(kv[1])):
        o.append("### %s / %s (planner %s, %d rows)\n" % (DATASETS[dskey]["name"], knob, direction, len(items)))
        ex = ", ".join("r=%.4f pred=%s tuned=%s" % (r, p, a) for r, p, a in items[:4])
        o.append("- rows: %s%s" % (ex, " ..." if len(items) > 4 else ""))
        o.append("- diagnosis: %s\n" % diagnose(knob, dskey, direction))
    small = {k: v for k, v in misses.items() if 0 < len(v) < 3}
    if small:
        o.append("\n## Non-systematic misses (<3 rows; listed, not diagnosed)\n")
        for (dskey, knob, direction), items in sorted(small.items()):
            o.append("- %s / %s (%s): %s" % (DATASETS[dskey]["name"], knob, direction,
                     "; ".join("r=%.4f pred=%s tuned=%s" % (r, p, a) for r, p, a in items)))
    o.append("\n## Knobs the laws genuinely cannot set yet (declared, not fit)\n")
    o.append("""- **Walk-mode R/B/W (round/frontier/width)**: P356's grid + H=E+(R-1)B give the accounting, not the values; the delta pilot is the bridge. Not scored.
- **hops beyond the band preset**: needs measured delta (recall-L4); mst h3-8 / WebVid h4-10 / t2i tail h4-6 are pilot territory.
- **Graph-free p and TFLOOR**: single anchor (cohere P304 ladder / P295 t=31.25p); no transferable constant.
- **TFLOOR p-coupling coefficient inside a band**: direction cited (jointly-scaled stages), constant dataset-specific (31.25p cohere, ~45p wiki, ~20p mst, flat DEEP).
- **GRAPH_M tail schedule**: 'M and p grow together' (P364) has no quantitative form; preset anchors 16/24/48 only.
- **SQ4 co-gate verdict**: containment pilot (GATE1-style) is a measurement; descriptor cannot predict cohere-10M's fail vs wiki's pass.
- **CASCADE_K containment escalation**: the P363 law names the branch (WebVid W>=32); the trigger is a measured containment loss.
- **RESIDENT at small footprints**: pays on DEEP (+27.6%), null on msturing -- pressure term missing (acknowledged law gap #2).
- **gamma value**: OOD on/off is law-set; the scalar (0.5 t2i / 0.2 WebVid) is a per-dataset calibration fit by design.
- **SQ4_INT8K constants**: cited per-regime defaults (256/512 d>=1024; 192/384 d=512); no scaling law.
- **nav 'lowrank' tier**: no law anchor exists; the planner never emits it.
""")
    o.append("\n## Honesty notes\n")
    o.append("""- The rung-within-band map (miss halves per work doubling) is the one place a modeling choice was needed; delta=0.5 is stated as an approximation of the measured 0.44-0.72 range (recall-L4), chosen a priori, not tuned to the rows.
- Where a systematic miss says PLANNER-vs-STALE-TUNING (fp16 tier), the claim is falsifiable: apply the lever to one non-DEEP/wiki frontier point and measure.
- Scoring counts msturing cascade_k disagreements even though P363 proved the knob inert there (W16==W256); see its diagnosis.
- Two ENGINE-vs-TUNED inconsistencies were surfaced rather than patched: the accurate d<=128 floor (1800 fits msturing, DEEP's tuned tail is flat 450) and the accurate 4.8k width (provision for binding cases; inert-wide on cohere/msturing). The planner mirrors the engine so the disagreement stays visible.
- Per-dataset COMMON flag tables are transcribed from tuned_optima.md's Notes blocks (stack lines), since flags live in prose, not the knob column.
""")
    with open(path, "w") as f:
        f.write("\n".join(o) + "\n")

# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrodict", action="store_true")
    ap.add_argument("--optima", default=OPTIMA)
    ap.add_argument("--out", default=SCORECARD)
    ap.add_argument("--d", type=int); ap.add_argument("--n", type=int)
    ap.add_argument("--metric", choices=["ip", "l2"], default="ip")
    ap.add_argument("--ood", choices=["in", "moderate", "extreme"], default="in")
    ap.add_argument("--qdensity", type=float, default=0.0, help="train-query density (fraction of base)")
    ap.add_argument("--graph-k", type=int, default=0); ap.add_argument("--ram-gb", type=float, default=256)
    ap.add_argument("--kf", type=int); ap.add_argument("--target", type=float)
    ap.add_argument("--band", choices=["frontier"])
    args = ap.parse_args()
    if args.retrodict:
        rows, per_knob, per_ds, systematic = retrodict(args.optima, args.out)
        tot = defaultdict(int)
        for c in per_knob.values():
            for k, v in c.items():
                tot[k] += v
        n = tot["exact"] + tot["step"] + tot["miss"]
        print("retrodicted %d rows; %d knob-fields scored: %d exact, %d step, %d miss (%s within-one-step)"
              % (len(rows), n, tot["exact"], tot["step"], tot["miss"], pct(tot["exact"] + tot["step"], n)))
        print("systematic miss groups: %d  -> %s" % (len(systematic), args.out))
        return
    if args.d is None or args.n is None:
        ap.error("plan mode needs --d and --n (or use --retrodict)")
    ds = dict(d=args.d, n=args.n, metric=args.metric, ood=args.ood, qdensity=args.qdensity,
              graph_k=args.graph_k, ram_gb=args.ram_gb, kf=args.kf)
    if args.band == "frontier":
        for t, p in plan_frontier(ds):
            print("== target %.3f ==" % t)
            print(json.dumps({k: v["value"] for k, v in p.items()}, indent=1, default=str))
    elif args.target is not None:
        p = plan(ds, args.target)
        for k, v in p.items():
            print("%-12s = %-28s # %s" % (k, json.dumps(v["value"], default=str), v["law"]))
    else:
        ap.error("give --target R or --band frontier")

if __name__ == "__main__":
    main()
