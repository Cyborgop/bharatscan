# Latency Budget

Update this file with measured numbers as they come in from `scripts/measure_device_latency.sh`.

## Target device: Snapdragon 6 Gen 1 (e.g. Redmi Note 12 Pro, Moto G73)

| Module | Precision | Input | Target (ms) | Measured (ms) | Notes |
|---|---|---|---|---|---|
| Corner detector | INT8 | 256×256 | 80 | TBD | NNAPI |
| Perspective warp | — | native | 60 | TBD | CPU / OpenCV |
| Doc classifier | INT8 | 224×224 | 40 | TBD | NNAPI |
| Text detection (DBNet) | INT8 | 640×640 | 350 | TBD | NNAPI, may fallback FP16 |
| Text recognition | INT8 | 48×320 × batch 8 | 700 | TBD | 20 avg lines, batched |
| Field extraction | — | — | 5 | TBD | Dart regex |
| **End-to-end** | — | — | **1235** | TBD | hard cap 2000 |

## Secondary devices

- Snapdragon 4 Gen 1 (budget, ~2 GB RAM): expect 1.8× slower → ~2.2 s. Acceptable if under 2.5 s.
- Snapdragon 7 Gen 1 (mid-premium): expect 0.7× → ~860 ms.
- Snapdragon 8 Gen 1 (premium): ~600 ms.

## Measurement protocol

1. Airplane mode ON (kill background network jobs)
2. Battery > 50%, not charging
3. 10 warmup runs, then 50 timed runs, report median + p95
4. Fresh device boot once per session
5. Same input image (`eval/test_sets/golden_aadhaar.jpg`)
