#!/usr/bin/env bash
# OOD 10M QPS@90% vs FLOAT GT — text2image, the real leaderboard scale.
# PREPPED, NOT auto-run: launch only when the box is clear (team-lead greenlights;
# heavy — 10M build + an 8GB float base + a ~9min float-GT build). Light otherwise.
#
# Stage 1 builds a 10M exact-float-IP GT (NQ=2000). Stage 2 sweeps the candidate
# operating point on BOTH paths (int8+t_surv-cut, float-rerank+t_surv-cut) vs that GT,
# so we can read off max QPS at recall@10>=0.90 and compare to ScaNN's published OOD QPS@90%.
set -euo pipefail
D=/home/thomas-ahle/lsh/big-ann-benchmarks/data/text2image1B
BIN=/home/thomas-ahle/lsh-engine-wt-ood2/sbann-rs/target/release/sbann
cd "$D"
NQ=${NQ:-2000}
THREADS=${THREADS:-8}

# --- Stage 1: 10M float GT (exact float IP top-10 over the first 10M float rows) ---
if [ ! -f t2i10m-floatgt ]; then
  echo "[10M] building float GT (NQ=$NQ, ~9min)..."
  RAYON_NUM_THREADS=16 "$BIN" floatgt base.1B.fbin.crop_nb_10000000 query.public.100K.fbin t2i10m-floatgt 10000000 "$NQ" 10
fi

COMMON="SBANN_IP=1 SBANN_FASTSCAN=1 SBANN_SOAR=0.5 SBANN_C0=256 SBANN_C1=4096 SBANN_B0=24 SBANN_B1=96 \
  SBANN_NQ=$NQ SBANN_REPS=5 SBANN_TFLOOR=1 RAYON_NUM_THREADS=$THREADS"
# p rescaled for 10M (~150 pts/cell at C=65536): sweep wide; t_surv kept SMALL (the cut).
PLIST=${PLIST:-512,768,1024,1536,2048}
TMUL=${TMUL:-3,4,6}

echo "=== INT8 + t_surv-cut, 10M vs FLOAT GT ==="
env $COMMON SBANN_PLIST="$PLIST" SBANN_TMUL="$TMUL" \
  "$BIN" run t2i10m.i8bin t2i10m_query.i8bin t2i10m-floatgt hierk3 apq4 3 65536 30

echo "=== FLOAT-rerank + t_surv-cut, 10M vs FLOAT GT (only survivors paged from the 8GB float base) ==="
env $COMMON SBANN_FLOAT_RERANK=1 SBANN_FBASE=base.1B.fbin.crop_nb_10000000 SBANN_FQUERY=query.public.100K.fbin \
  SBANN_PLIST="$PLIST" SBANN_TMUL="$TMUL" \
  "$BIN" run t2i10m.i8bin t2i10m_query.i8bin t2i10m-floatgt hierk3 apq4 3 65536 30
