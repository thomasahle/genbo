# HANDOFF (2026-07-07, post-P252)

## Standing: ALL benchmark fronts won or closed
| front | verdict | champion config |
|---|---|---|
| t2i-1M OOD | WON 0.755x (1t) / 0.746x (8t) vs ScaNN | eng_t2i1m_kf16384_c768_b96_a3_em2 + γ0.5 + graph M25 t470/p18 |
| t2i-10M OOD | WON 0.827x (1t) / 0.758x (8t) | eng_t2i10m_kf65536_c4096_b128_a3_em2 + γ0.5 + graph M32 t1000/p40 |
| t2i-100M OOD | ~6700-6780 QPS@0.90 (16t); ScaNN build-INELIGIBLE @2h gate | eng_t2i100m_kf524288_c2048_c1_32768_b512_a3 (em0) + γ0.5 + graph M64 t8000/p76 |
| cohere-1M in-dist | WON vs HNSW: 0.9220@3229 vs 0.9122@1840 (1t) | eng_cohere1m_kf16384_dpb4 + graph M16 + int16 policy |
| cohere-10M in-dist | WON vs HNSW: 1t median 1.098 (0.9390@~1945 vs 0.9370@~1672); 8t 12850 vs ~11930 | eng_cohere10m_l2_65536_dpb4_em3_a3 + graph M16 t400/p16 (needs BATCH_CHUNK auto = P246 binary) |
| streaming 30M | CLOSED at compute frontier: 0.8821@52.7min eligible (this box); machine-relative | feat/streaming-30m worktree @4173d70 |
| RoarGraph (OOD SOTA) | measured 1469 QPS@0.903 t2i-10M 1t — 4-5x behind genbo | ~/RoarGraph, index built, search cmd in P-notes |

## This session's engine changes (all on champion, flag-gated / bit-identical to prior champion paths)
- P239 scan-precision policy: auto int16 LUT when m>128 (f2a99fc) — THE in-dist unlock (d=768 was 0.07 in-pool ordering under int8-sat)
- P241 avx512-i16 pair-scan gate hoisted to startup AtomicBool, default on (187c0ac): +31% cohere-10M
- P245/P246 adaptive batch-chunk clamp(nq/(4*threads),125,1000) (6724117): fixes nq<=1000 serialization + stragglers; +41% t2i 8t
- P251/P252 ROUTE_SDIM0 + packed-prefix routing centroids (adf0ff4, 0dbd96c): route 285->98us on rotated basis; frontier-dependent (rotation's int8-scale cost); champion unchanged
- d>=768 support fixes: pq/vq/simd buffer bumps (earlier this session)

## Ledger: FINDINGS.md current through P252 (paper_ood.tex + PDF pushed; distribution-robust reframe done)

## RUNNING (all detached, marker-gated):
- wiki-35M download (CohereLabs/wikipedia-2023-11 en, d=1024, cap 35M, RESUMABLE): logs/wiki35m_download2.log, marker WIKI35M_DOWNLOAD_DONE (~17M/35M, HF rate-limited)
- prep chain (queries+exact GT+int8 per-basis scale): logs/wiki35m_prep.log, marker WIKI35M_PREP_DONE
- master chain (genbo build [8192,262144] dpb4 em3 a0=3 || HNSW build+graph -> h2h): logs/wiki_chain.log, marker WIKI35M_CHAIN_DONE
- GOTCHAS for wiki-35M: d=1024 verified no-panic (m=512 & 256); int8 scale MUST be per-basis (P250); pkill-self (use kill-by-PID)

## Open threads (none blocking)
- 12h-gate scann-100M h2h (optional; 2h-gate ineligibility banked)
- 1B ambitions (int8 base 200GB fits RAM; days of setup)
- rotation without int8-scale cost (per-block scale / route-only rotation) would make packed-SDIM a strict win (P252)
- paper: neurips class swap + authors at submit time

---
# SESSION-2 ADDENDUM (2026-07-08, P253–P262)

**Everything below is on `champion`; ledger FINDINGS.md current through P262; paper_ood.tex reframed distribution-robust, compiles, PDF pushed.**

## New engine features (all flag-gated; champion default bit-identical)
- **SBANN_GRAPH_HOPS** (P253/P262): R-round batched graph-repair walk (expand best-M → rescore → reselect). Default 1. A **high-recall lever** (matched-recall win only above ~0.91–0.92 crossover; at the 0.90 gate hops=1 is ~6% faster). Already works in the batched 8t driver (shares rerank_cascade_graph). Theory: miss(R,p)=m_inf(p)+A(p)·δ^R, δ≈0.44, R²=0.99 (P255) → no optuna, tune p then R.
- **SBANN_GRAPH_BESTFIRST** (P256/P260): global-best-first frontier. +0.3–0.8pt IN-DISTRIBUTION only; neutral OOD (base-metric edges ⊥ off-manifold queries). Opt-in.
- **SBANN_ROUTE_SDIM0 + packed-prefix routing centroids** (P251/P252): route 285→98µs on rotated basis; parity with unrotated at 0.93 (rotation's int8-scale cost offsets); frontier-dependent, champion unchanged.
- **SBANN_BEAM0** (P259): query-time coarse-beam override (no rebuild to widen coverage).

## Verdicts
- **Multi-hop line CLOSED (P262)**: real+useful but correctly scoped (high-recall, not gate); headline OOD ratios (1M 0.755 / 10M 0.827) UNCHANGED, champion stays hops=1.
- **wiki-35M CLOSED (P261)**: honest method frontier. genbo wins in-dist ≤10M/d768 (cohere) but loses at 35M/d1024 (Cohere-v3 wiki, ‖μ‖=0.48) — routed-4bit-PQ can't resolve tight in-dist top-k at scale; HNSW-int8 wins (0.967@1198 1t). Index verified correct (self-query). genbo build FASTER (1.9h vs 3.8h). Future: finer/higher-bit PQ or query-time exactness stage for tight in-dist top-k.
- **RoarGraph (P254)**: tight pairwise genbo WINS t2i-10M OOD median 1.07× at matched 0.90 (banked solo 1469 was cold-cache).

## Gotchas re-learned
- Compare at the recall GATE, not fixed p (fixed-p "wins" are mirages — P262, cf P217).
- Long detached jobs need **setsid** (nohup dies to tool-timeout SIGTERM); kill-by-PID (pkill-self).
- faiss HNSW fp32 at 35M×1024 OOMs (271GB) → use IndexHNSWSQ fp16/int8 (matched to genbo's memory budget).

## Open / next
- wiki-35M finer-PQ fix (deferred); paper: neurips class + authors at submit; 1B (int8 200GB fits, blocked on ~800GB download + GT + scann-1B baseline).
