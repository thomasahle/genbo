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
