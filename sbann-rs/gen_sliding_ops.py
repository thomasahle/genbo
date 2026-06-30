#!/usr/bin/env python3
"""Emit a sliding-window streaming .ops file for the `stream_runbook` subcommand, sized to a base of
N distinct vectors (so it replays faithfully on our 1M msturing base — orig id == base row, every id
< N). Mirrors the NeurIPS-23 msturing sliding-window pattern (fill a window, then on each step delete
the oldest chunk + insert the next chunk + search) but bounded to N total points so no vector is reused.

Usage: gen_sliding_ops.py [N] [W] [S]   (defaults 1_000_000 / 500_000 / 50_000)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
W = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
S = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000

lines = [f"max_pts {N}"]
# fill the window in S-chunks
g = 0
while g < W:
    lines.append(f"insert {g} {min(g + S, W)}")
    g += S
lines.append("search")                       # steady state reached, window full
# slide: delete oldest chunk, insert next chunk, search — until we exhaust the N points
g = W
while g + S <= N:
    lines.append(f"delete {g - W} {g - W + S}")
    lines.append(f"insert {g} {g + S}")
    lines.append("search")
    g += S
sys.stdout.write("\n".join(lines) + "\n")
