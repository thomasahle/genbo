# HANDOFF — big-ann leaderboard push (OOD + streaming, 1M/10M/100M)

Written 2026-07-04, updated same day after the GitHub push. Previous agent: Claude session 06297076 (loop-driven, ~P185→P218 in FINDINGS.md).

## GITHUB (added 2026-07-04)
The project is now **`github.com/thomasahle/genbo`** ("genbo" = Danish for the neighbor across the street — cross-modal OOD search). 29 refs pushed: default branch **`champion`** (canonical engine), `aniso-vq-faithful` (FINDINGS ledger + this handoff), `feat/streaming-30m`, `master` (theory), and ~20 experiment branches (the provenance for every ledger claim).
- Remote name: `genbo` (configured in the `lsh-engine` worktree; the underlying repo is `/home/thomas-ahle/lsh`, of which all `lsh-engine*` dirs are worktrees).
- Auth: this box's SSH key and `gh` token are the **`thomasnormal`** account, a collaborator on the repo (the owner is `thomasahle`).
- **Push discipline**: after committing FINDINGS entries or landing branch work, `git push genbo <branch>` — the remote is the only off-box copy of the ledger.
- **Visibility**: the repo was PUBLIC at push time (user aware; check current setting before assuming).

## THE GOAL (user's standing directive)
"Keep innovating, iterating and improving until we top the leaderboard for OOD and streaming. Both 1M, 10M and 100M."
Two tracks: **OOD** = big-ann text2image (d=200, IP, float GT) and **STREAMING** = msturing-30M-clustered final_runbook (d=100, L2, 8GB/1hr caps).
Hard constraint from the user: **never use more than ~40% of system memory** (64GB box → stay under ~26GB; the box is SHARED — other tenants run heavy chip-design builds; load 20–80 is normal).

## SOURCES OF TRUTH (read these before doing anything)
1. **`/home/thomas-ahle/lsh-engine/FINDINGS.md`** — the canonical experiment ledger, entries P1–P218 (newest just before the `=== SESSION SUMMARY` marker). Every result, refutation, and gotcha is there. Append new entries the same way (python heredoc insert before the marker + git commit on branch `aniso-vq-faithful`, which is what the main `lsh-engine` worktree has checked out).
2. **Claude memory dir** `/home/thomas-ahle/.claude/projects/-home-thomas-ahle-lsh/memory/` — `ood-10m-standing.md`, `streaming-track-result.md`, `clean-abstractions-requirement.md` hold the compressed current state.
3. **Scratchpad** `/tmp/claude-1013/-home-thomas-ahle-lsh/06297076-ad84-405e-a162-335c716d585e/scratchpad/` — all harness scripts, logs, indexes, CSVs. Key scripts named below.

## CURRENT STANDING

### OOD 1M — WON (measured), consolidated
- **0.91× vs ScaNN** (we are ~9.5% FASTER), directly measured: 8-round interleaved decider, in-round ScaNN, calm box (load 17–20), recall id-recomputed (ours 0.9060 vs ScaNN 0.9032). FINDINGS **P216**. HANNS (leaderboard #1) leads ScaNN by only ~7% ⇒ this is leaderboard-top-equivalent at 1M single-thread.
- Winning stack ("GR18_t470"): **γ=0.5 routing calibration** (the dominant lever — score fine cells by `γ‖c‖²−2q·c` as a per-cell additive bias, zero query cost; P206) + **kNN-graph pool expansion** (k=16 base-side sidecar, top-M=25 pool members expanded, union int8-rescored; P205/P207) + **fused-union trim** (P214) + t_surv=470, p=18, cascade K=16 — all on top of the earlier primitives (FastScan2, route-VNNI, cascade K16, fused-topk, cell-major batched scan P202, prefetch).
- **Consolidated: branch `champion` @ `4aa59af`** (worktree `lsh-engine-wt-champion`), all gates passed, champion-default recall 0.9032 unchanged (γ default-unset = bit-identical; graph flag-gated `SBANN_GRAPH_FILE`; recipes in the "OOD CALIBRATION STACK" doc block above `main()`). FINDINGS **P218**.
- The mechanism insight that cracked it: the OOD gap was **routing miscalibration** (L2 ordering over-penalizes large-norm centroids that IP favors), NOT the codebook. All code-side levers were correctly refuted (P182–P201: anisotropy ×3, richer codes, NormPq, OPQ, SoA relayout, ADC routing P210, cascade geometry re-tune P211).

### OOD 10M — γ proven, honest h2h STILL OPEN (the main in-flight item)
- γ transfers: **2.5–2.7× probe cut at matched recall** (P215). t_surv must scale with n (540@1M → 2000@10M; too-shallow pools cap recall at ~0.88 regardless of p).
- Engine 10M index built: `eng_t2i10m_kf131072_c4096_b256_a3.idx` (scratchpad, 7.9GB). Engine arm: γ=0.5, t_surv=2000, p=45 → 2512 QPS @ 0.9054 (loaded).
- **A 0.505× ratio vs ScaNN was measured but is QUARANTINED (P217)**: the cached ScaNN 10M index has num_leaves=4000 vs ScaNN's official ~40000 — under-leaved, flatters us. **The corrected measurement (ScaNN rebuilt @ 40k leaves) has failed ~4 times and is NOT currently running** (verified 2026-07-04: chain logs frozen at Jul 2 16:24, the `chain10m` tmux session holds a dead shell, zero scann processes). Every attempt died during ScaNN's hashing phase (tenant load spikes / session crashes; details under "infrastructure lessons"). To relaunch: `tmux kill-session -t chain10m; tmux new-session -d -s chain10m "bash <scratchpad>/chain_10m_retry.sh"` — the wrapper waits for load<35 & avail≥15GB, retries 3×. Stages: build (`scann_build_40k.py`, ~17GB RSS transient, ~10–30 min) → lts sweep (`scann_measure10m.py`) → 20-round pairwise. Success marker `ALLDONE_CHAIN10M`; results land in `h2h_10m_40k.{log,csv}`. Babysit stage 1 — it is the fragile part.
- Expectation: corrected ratio likely lands between 0.6× and 1.0× (our fine-partition scaling advantage is mechanistically real, but 4k→40k leaves cuts ScaNN's per-probe scan ~10×).

### STREAMING 30M — WON (fully eligible), banked
- Official msturing-30M final_runbook: **avg recall@10 = 0.9654, peak 5.32GB < 8GB, ops wall 53 min < 1hr** — all gates green *under load ~35* (P212). Timing question settled (P213): the 1hr container clock includes train, but train is only 27s (the earlier 997s scare was our own inline GT eval, which the official harness does offline). **2nd open-source tier**: pyanns 0.9597 < OURS 0.9654 < hwtl-closed 0.9675 < puck 0.9855/0.9849.
- Branch **`feat/streaming-30m` @ `c1c24e3`** (worktree `lsh-engine-wt-streaming2`): f16 rerank cache (lossless, halves the float cache — this is what fits 8GB), bounded-SOAR-spill insert fix (3.1k→26k inserts/s, recall-identical), glibc arena trim (peak 8.64→5.32GB).
- Pre-submission compliance note: cold-start currently trains the router on a disk base-sample (peeks at future data). Clean fix: train on the first insert batch (27s, negligible). Do before any official PR.

### NOT YET TOUCHED
- **100M** on either track (memory: 100M t2i int8 = 20GB — needs a quiet box and careful budgeting against the 26GB ceiling).
- **Multi-thread / 8-vCPU** measurements (the official leaderboard protocol). Our per-query Rust loop historically suffers oversubscription more than ScaNN's batched C++ (P197 note); the P202 cell-major batched driver may change that — unmeasured.
- Streaming recall push toward puck's 0.9855: the tail is low-live steps (<0.5M live: 0.868); lever = adaptive-p by live density + the ~7min quiet time-budget margin.

## MEASUREMENT METHODOLOGY (non-negotiable; this discipline caught every mirage)
1. **Same-hardware only.** Never compare our QPS to leaderboard numbers from other machines (the original "25× behind" was exactly that error; the true gap was 2.22×).
2. **Interleaved, same-window, in-round opponent.** ScaNN 1.4.2 lives in `/home/thomas-ahle/scann_venv`. Single-thread pinned (`taskset`), best-of-5 per arm per round, 6–8+ rounds, median. On a loaded box use **tight pairwise alternation** (adjacent measurements seconds apart, median of paired ratios — tolerates load drift; block best-of-5 doesn't).
3. **Recall gates are exact**: recompute recall@10 from raw result ids vs the float GT (never trust the harness's cached number for a headline); pick operating points on one run, CONFIRM on a fresh run; never cherry-pick.
4. **The opponent runs at its best.** ScaNN gets its official recipe (1M: tree 2000 + AH2 thresh 0.2 + reorder ~78–200, `search_batched`; 10M: ~40000 leaves). Any win that depends on a handicapped baseline gets quarantined (see P217).
5. Absolute QPS on this box is noise; ratios are the deliverable. Loaded ratios were historically PESSIMISTIC for us pre-batching; with both engines batched they're roughly fair, but calm-window (load <18–20) numbers are the bankable ones.

## INFRASTRUCTURE LESSONS (hard-won; will bite you otherwise)
- **Process survival**: anything launched from your Bash tool — even `setsid nohup` — dies with your session (cgroup teardown). For runs that must survive, launch inside the user's tmux server (`tmux new-session -d -s <name> "cmd"`). Check `tmux ls` first.
- **Memory kills are silent**: three long runs died without any log line (no panic, no dmesg OOM visible). Wrap long runs with a supervisor that records exit status, and Monitor log files for BOTH success markers AND process death.
- **Don't run >2 concurrent cargo/release builds**: 7 parallel builds once drove the box to fork-failure (load 65+, `ld` SIGABRT, rustc ICE). Builds: `RAYON_NUM_THREADS=3-4`, link with `-j2` if flaky.
- **ScaNN 40k build specifics**: ~17GB RSS transient (use `np.fromfile`, NOT `read()+copy` — that was 16GB extra and got OOM-killed); partitioner ~5 min; the hashing phase is the fragile part.
- Idle background agents don't wake themselves; if you orchestrate agents, nudge them via SendMessage when their background jobs finish, and have them CHAIN stages inside one wrapper script instead of idling between stages (`chain_soar.sh` pattern).
- Gotcha that keeps recurring: the champion operating points use **fixed t_surv via `SBANN_TFLOOR` (540@1M, 470 graph-mode, 2000@10M), NOT p·TMUL** — using tmul makes γ arms look broken (0.87 recall).

## SUGGESTED PRIORITY ORDER FOR THE NEXT AGENT
1. **Finish the corrected 10M h2h** (check the tmux `chain10m` attempt first). Bank the honest ratio as the P217 resolution. If >1×, decompose phases (route/scan/refine vs ScaNN) — γ+graph composition at 10M (graph sidecar would need a 10M kNN build, ~30–60 min via engine self-query) is the ready lever.
2. **Multi-thread batched measurements** (8 threads, both engines, 1M then 10M) — the actual leaderboard protocol. Needs a relatively calm box.
3. **100M OOD**: int8 base 20GB → build only when box has ~30GB+ available; consider f16/int8-only representations; Kf scaling per P211's total-cells-scored logic (~2500–2800 cells *scored* stays the invariant, t_surv scales with n).
4. **Streaming toward puck**: adaptive-p on low-live steps; then the 100M/1B streaming variants if the track offers them.
5. Housekeeping: 16+ worktrees exist (`git worktree list`); the live ones are `lsh-engine-wt-champion` (canonical OOD), `lsh-engine-wt-streaming2` (streaming), `lsh-engine-wt-probecal`/`-graphimpl` (history of the winning levers). Others are banked experiment branches — keep for provenance, don't build on them.

## KEY PEOPLE/CONVENTIONS
- User: Thomas Ahle (expert; wants honest numbers over good news — every mirage correction came from his prodding: same-hardware, ADC-first, "you can't compare across machines"). Keep the codebase clean per `clean-abstractions-requirement.md`: quant methods behind the Compressor trait, hardware behind runtime `is_x86_feature_detected!` dispatch with scalar fallback, env flags only as temporary A/B scaffolding, consolidation passes when levers stabilize.
- Ledger discipline: every experiment gets a P-entry (including refutations — they're the map). The arc 25×→2.22×→1.77×→1.45×→1.20×→0.97×→0.91× exists because refuted paths were recorded and never re-run blind.
