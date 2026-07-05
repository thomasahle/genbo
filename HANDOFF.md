# HANDOFF — big-ann leaderboard push (OOD + streaming, 1M/10M/100M)

Updated 2026-07-05 (supersedes the 2026-07-04 handoff, which described the OLD 16-core/64GB box — that
environment is gone). Ledger: FINDINGS.md P1–P227, canonical on branch `champion` in THIS repo (~/genbo).

## GOAL (user's standing directive, /goal-hooked)
Beat all SOTA ANN methods (ScaNN, HANNS bar = ScaNN×1.07 ⇒ ratio ≤0.93) on the big-ann datasets ON THIS
MACHINE. Tracks: OOD (text2image, d=200 IP, float GT) and STREAMING (msturing-30M-clustered final_runbook).

## THIS BOX
96 cores / 371GB, shared with EDA tenants (load 15–35 typical). **CPU etiquette (user-set): builds ≤8–16
nice'd cores, pinned via `taskset -a -pc`; measurements single-core (or 8-core for the 8-thread protocol).**
Per-core it is SLOWER than the old Zen4 box (~0.7× scann-1M single-thread; ~2–3× slower for the streaming
runbook at 8 threads — this matters for streaming eligibility only).
Gotcha: never pkill/pgrep a pattern contained in your own command line (kills your own shell, exit 144);
list PIDs first, kill by literal PID separately. tmux commands: absolute paths only (session cwd varies).

## RIG (all under /home/thomas-ahle/big-ann-data/)
- t2i data: base{1M,10M,100M}.fbin + .i8bin (shared scale = 127/max|base|: 330.19 / 300.32 / 299.31),
  query.public.100K.fbin (+ per-scale i8bin), exact float-IP GTs t2i{1m,10m,100m}_gt.ibin (10k q × top-100,
  ScaNN-brute-force computed, numpy spot-checked).
- ScaNN 1.4.2 in ~/scann_venv (py3.11). OFFICIAL OOD searcher rebuilt from the VERBATIM textproto
  (big-ann-benchmarks neurips23/ood/scann; the GCS pre-built one is no longer public): scann_t2i10m_40k/
  (40k leaves), scann_t2i1m_2k/ (1M, ledger convention), scann_t2i100m_126k/ building (126k leaves, 20M
  training sample — no official 100M recipe exists; leaves ∝ √n extrapolation, noted deviation).
- Engine indexes: eng_t2i1m_kf16384_c768_b96_a3_em2.idx, eng_t2i10m_kf65536_c4096_b128_a3_em2.idx
  (champion), G0/G1/G3/G4 variants (geometry bracket), eng_t2i100m_kf524288_c32768_b128_a3_em2.idx building.
- Graph sidecars: t2i{1m,10m}_graph_k16.u32 (ScaNN self-search); 100M via the new `selfknn` subcommand
  (engine self-query, 92.7% edge agreement with scann-built at 1M, 27s/1M @8thr).
- Harness: h2h_10m_pairwise.py (env-parametrized: GT/SCANN_DIR/BASE_I8/QUERY_I8/FBASE/ENG_IDX,
  H2H_THREADS for the 8-thread protocol), scann_measure10m.py (same envs), pristine_confirm.sh
  (SIGSTOP-builds confirmation pattern). Streaming: ~/genbo-streaming worktree (feat/streaming-30m),
  data+639 step-GTs in streaming/, final_runbook.ops (converter recreated inline).

## RESULTS ON THIS MACHINE (opponent at its official best, same exact float GT, recall gates ≥0.90 held,
## tight-pairwise warm protocol, median of per-round ratios)
- **OOD 1M: WON 0.755× (1t) / 0.756× (8t)** — P221/P222. Engine arm: γ0.5 + graph M25 + t470/p18.
- **OOD 10M: WON 0.827× pristine (1t) / 0.833× (8t)** — P224–P227; arc 1.239→1.009→0.924→0.847→0.827.
  Champion arm: Kf=65536 C0=4096 b0=128 a0=3 SOAR EM(2,beam8) + γ0.5 + graph(M32,kedge16) + t1000/p40/K16
  + float rerank. **Geometry knee = 152-pt cells** (G3 finer and G4 coarser both lose — with graph-union +
  float-rerank, coarse cells beat the fine-cells intuition; evals/query 6144).
- **OOD 100M: in flight** — engine + scann builds running; post-build chain queued (sweep → selfknn graph →
  graph sweep → h2h). t_surv scaling expectation: ~2000→3000-4000 at 100M.
- **STREAMING: compliant recall 0.9745 (NQ=100) / 0.9706 (NQ=1000)** vs banked 0.9654 — P226. Compliance
  fixed (first-insert-batch cold-start + SBANN_RETRAIN_EVERY geometric live-set retrains, ~free) and the
  low-live tail killed (SBANN_RB_MINCAND adaptive-p: worst step 0.737→0.944). Peaks ≤7.1GB < 8GB.
  Recall ladder: p64→0.881, p96→0.914, p128/t512→0.934, p160/t640→0.948, p256/t1000→0.9706.
  OPEN: the 1-hour wall on THIS box (2-3× slower than Azure-class at 8thr) cuts the ladder around
  p≈128-160; definitive NQ=10000 runs queued (p160/t640 + p256/t1000). Report BOTH this-box-eligible and
  Azure-projected numbers (old-box P212 evidence: this stack class fits easily there).

## KEY ENGINE CHANGES THIS SESSION (committed)
- champion: parallel-chunk batched driver (the serial chunk loop silently defeated RAYON — 8t was 1t;
  now 7.6× at 8t, chunk≈250 at 8 threads, bit-identical recall). `selfknn` subcommand. Ledger P219–P227.
- feat/streaming-30m: compliance retrain + adaptive-p (2cdc7d1), SBANN_TFLOOR knob for stream_runbook.

## METHODOLOGY (unchanged, non-negotiable — it caught every mirage again this session)
Same-hardware only; tight pairwise alternation, order alternating, best-of-5, median of paired ratios;
warm-resident both arms (leaderboard semantics — the v1 cold-spawn asymmetry penalized the engine; the
pristine pattern = SIGSTOP all own builds); recall recomputed vs the shared exact float GT; opponent at
its official config with its own operating-point sweep (incl. corners: high-lts/low-reorder checked and
rejected). Recall is the only load-independent column — never compare QPS across runs while builds churn
(25% swings observed).

## REMAINING FOR THE GOAL
1. 100M OOD h2h (builds + queued chain; then scann-100M lts sweep + pairwise h2h, pristine pattern).
2. Streaming: definitive NQ=10000 runs → bank this-box-eligible + headline numbers. Recall push toward
   puck 0.9855 beyond that likely needs insert-speed work (inserts eat 50%+ of the budget here).
3. Claim hygiene: HANNS is binary-only — the ≤0.93 bar is the operationalization; other open runnable
   contenders (pyanns etc.) rank far below ScaNN on OOD, covered by transitivity, but could be run here
   for completeness if the user wants.
