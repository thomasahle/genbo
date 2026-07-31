# Uniform all-dataset frontier campaign

Full official query sets, one thread, best of five repetitions. Each artifact family receives one highest-work full-query warmup, then an exact forward/reverse plan in one loaded process. A run is accepted only when recall matches in both directions and every mirrored QPS pair differs by at most 5%. Curves use the latest accepted run per family; no points are stitched across windows within a family.

## Materialized envelopes

| dataset | points | recall range | families |
|---|---:|---:|---|
| cohere10m | 5 | 0.9461--0.9912 | hybrid |
| cohere1m | 6 | 0.9732--0.9971 | cascade |
| deep10m | 21 | 0.8041--0.9980 | cascade, walk |
| mst30m | 8 | 0.7268--0.9883 | cascade |
| t2i100m | 5 | 0.8796--0.9233 | nd16 |
| t2i10m | 18 | 0.8456--0.9954 | co16, hybrid, nd16, nd32, ndk64 |
| t2i1m | 26 | 0.7984--0.9914 | kf16384-hybrid, kf16384-nd16, kf2048-hybrid, kf4096-co16, kf4096-hybrid, ndk64 |
| webvid | 16 | 0.8062--0.9851 | seed128-k64, seed256-k64, seed512-k32, seed512-k64, seed64-k64, seed768-k32, seed768-k64 |
| wiki35m | 13 | 0.9328--0.9925 | cascade |

DEEP's fixed-round walk and cascade are emitted as separate CSVs and are not visually connected. Its legacy 0.9987 point is absent: the current corrected-fp16/layout stack plateaus at reproduced recall 0.9980 for p=768--1024.
