#!/usr/bin/env python3
"""Convert a NeurIPS-23 big-ann STREAMING runbook YAML into a flat .ops file that the
Rust `stream_runbook` subcommand consumes (no YAML dep in the engine).

Usage:
  runbook_to_ops.py <runbook.yaml> <dataset_key> [--scale-to N]

Emits to stdout, one op per line:
  max_pts <N>
  insert <start> <end>
  delete <start> <end>
  search
  replace <tags_start> <tags_end> <ids_start> <ids_end>

--scale-to N rescales every point index by N/max_pts (and sets max_pts=N), so a 5M/10M
runbook can be replayed faithfully at a smaller base (e.g. our 1M msturing) while keeping
the insert/delete/search interleaving and the sliding-window steady state intact.
"""
import sys, yaml


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, key = sys.argv[1], sys.argv[2]
    scale_to = None
    if "--scale-to" in sys.argv:
        scale_to = int(sys.argv[sys.argv.index("--scale-to") + 1])

    rb = yaml.safe_load(open(path))[key]
    max_pts = rb["max_pts"]
    out_max = scale_to if scale_to is not None else max_pts
    sc = (lambda i: min(out_max, round(i * out_max / max_pts))) if scale_to is not None else (lambda i: i)

    lines = [f"max_pts {out_max}"]
    i = 1
    while i in rb:
        e = rb[i]
        op = e["operation"]
        if op in ("insert", "delete"):
            a, b = sc(e["start"]), sc(e["end"])
            if b > a:                       # drop empty ranges produced by rounding
                lines.append(f"{op} {a} {b}")
        elif op == "search":
            # emit the 1-based ORIGINAL runbook op index: the official per-step GT is named
            # step{i}.gt100 (download_gt.py uses enumerate over the full op list), so the eval
            # loads the right GT regardless of any dropped/scaled ops above.
            lines.append(f"search {i}")
        elif op == "replace":
            lines.append(f"replace {sc(e['tags_start'])} {sc(e['tags_end'])} {sc(e['ids_start'])} {sc(e['ids_end'])}")
        else:
            sys.exit(f"unknown op {op}")
        i += 1
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
