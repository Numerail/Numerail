# Numerail Performance Benchmark Report

Generated: 2026-03-27 16:26:55 UTC

## Platform

- **Python**: 3.11.9
- **Platform**: Windows-10-10.0.26200-SP0
- **CPU**: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
- **NumPy**: 1.26.0
- **Runs per benchmark**: 1000 (first 100 discarded as warmup)
- **numerail_ext available**: yes

## Executive Summary

- **Total benchmarks**: 72
- **Fastest operation**: `2h. BudgetTracker.rollback()` — 0.45 µs mean
- **Slowest operation**: `4h. governor.enforce_next_step() — CLOSED (healthy)` — 133282.3 µs mean
- **Median mean latency across all benchmarks**: 77.3 µs
- **Median p99 latency across all benchmarks**: 136.2 µs

### Key Throughput Numbers

- **NumerailSystemLocal full stack (2 fields)**: 6,214 ops/sec
- **Peak throughput (fastest path)**: 2,223,210 ops/sec (`2h. BudgetTracker.rollback()`)

### Deployment Latency Corridor

Estimated end-to-end latency budget per enforcement call in a
production deployment (p99 components, additive model):

| Component | p99 µs | Notes |
|-----------|-------:|-------|
| 4D approve (policy load cached) | 50.7 | — |
| 8D project (correction path) | 400.6 | — |
| Audit record append | 21.5 | — |
| Budget delta write | 2.7 | — |
| Full local stack (2 fields) | 258.3 | — |
| Full local stack (8 fields) | 308.1 | — |
| **Estimated full-stack p99** | **1041.9** | Additive model |

*Note: real production adds network, DB transaction, and outbox overhead.*

### Bottleneck Analysis

Top 10 highest p99 operations:

| Rank | Benchmark | p99 µs | mean µs |
|------|-----------|-------:|--------:|
| 1 | 4h. governor.enforce_next_step() — CLOSED (healthy) | 139430.2 | 133282.3 |
| 2 | 6c. AuditChain.verify() — 1000 records | 11005.8 | 6779.9 |
| 3 | 1l. enforce 8D linear+quadratic (PROJECT) | 7462.3 | 6072.4 |
| 4 | 2k. AuditChain.verify() 500 records | 4514.9 | 3210.4 |
| 5 | 6c. AuditChain.verify() — 500 records | 4371.1 | 3348.8 |
| 6 | 4j. contract.verify_digest() | 1934.3 | 1181.9 |
| 7 | 6c. AuditChain.verify() — 250 records | 1933.1 | 1636.5 |
| 8 | 6b. enforce — 8D 64D linear (PROJECT) | 1801.4 | 1411.9 |
| 9 | 4i. NumerailPolicyContract.from_v5_config() | 1762.9 | 1203.7 |
| 10 | 6b. enforce — 8D 30D linear (PROJECT) | 1118.4 | 658.0 |

**Lowest-latency operations (fast path):**

| Rank | Benchmark | mean µs | ops/sec |
|------|-----------|--------:|--------:|
| 1 | 2h. BudgetTracker.rollback() | 0.45 | 2,223,210 |
| 2 | 4a. BreakerStateMachine.update() — CLOSED (healthy) | 1.05 | 952,744 |
| 3 | 4b. BreakerStateMachine.update() — THROTTLED (high load) | 1.14 | 877,963 |
| 4 | 2f. Schema.devectorize() 8 fields | 1.24 | 809,651 |
| 5 | 6d. BudgetTracker.record_consumption() — 1 budgets | 1.37 | 732,172 |

## Detailed Results

All latencies in **µs (microseconds)**. Timing: `time.perf_counter_ns()`.

### 1. Core Enforcement Latency

| Benchmark | mean µs | median µs | p95 µs | p99 µs | min µs | max µs | ops/sec |
|-----------|--------:|----------:|-------:|-------:|-------:|-------:|--------:|
| 1a. enforce 2D box (APPROVE) | 24.51 | 23.80 | 27.70 | 42.00 | 23.00 | 59.80 | 40,799 |
| 1b. enforce 4D linear (APPROVE) | 26.08 | 25.00 | 32.20 | 50.70 | 24.20 | 63.40 | 38,339 |
| 1c. enforce 4D linear (PROJECT) | 182.48 | 180.10 | 194.50 | 233.70 | 176.90 | 268.40 | 5,480 |
| 1d. enforce 4D linear (REJECT mode, infeasible) | 20.09 | 19.80 | 21.00 | 26.60 | 19.10 | 46.50 | 49,778 |
| 1e. enforce 8D linear (APPROVE) | 26.79 | 26.20 | 28.00 | 40.40 | 25.50 | 94.90 | 37,329 |
| 1f. enforce 8D linear (PROJECT) | 254.62 | 247.60 | 285.90 | 400.60 | 242.00 | 683.50 | 3,927 |
| 1g. enforce 16D linear (APPROVE) | 29.81 | 27.90 | 45.90 | 59.00 | 27.20 | 119.50 | 33,546 |
| 1h. enforce 16D linear (PROJECT) | 469.14 | 460.00 | 508.70 | 710.10 | 452.70 | 941.80 | 2,132 |
| 1i. enforce 30D linear (APPROVE) | 32.55 | 32.10 | 33.50 | 46.30 | 31.10 | 62.80 | 30,719 |
| 1j. enforce 30D linear (PROJECT) | 645.97 | 634.60 | 709.90 | 900.10 | 622.60 | 1058.70 | 1,548 |
| 1k. enforce 8D linear+quadratic (APPROVE) | 38.18 | 37.60 | 40.80 | 52.30 | 36.40 | 94.20 | 26,191 |
| 1l. enforce 8D linear+quadratic (PROJECT) | 6072.40 | 5980.10 | 6591.20 | 7462.30 | 5825.20 | 8894.80 | 165 |

### 2. Individual Function Latency

| Benchmark | mean µs | median µs | p95 µs | p99 µs | min µs | max µs | ops/sec |
|-----------|--------:|----------:|-------:|-------:|-------:|-------:|--------:|
| 2a. FeasibleRegion.is_feasible (feasible) | 5.26 | 5.20 | 5.30 | 5.80 | 5.00 | 30.10 | 189,937 |
| 2b. FeasibleRegion.is_feasible (infeasible) | 5.99 | 5.30 | 8.20 | 11.40 | 5.10 | 45.80 | 166,820 |
| 2c. project() box 8D (fast convergence) | 44.87 | 44.10 | 48.80 | 59.70 | 43.20 | 89.50 | 22,288 |
| 2d. project() linear 8D | 577.16 | 567.40 | 612.90 | 811.10 | 556.40 | 1062.90 | 1,733 |
| 2e. Schema.vectorize() 8 fields | 6.25 | 6.20 | 6.30 | 6.40 | 6.10 | 9.10 | 160,126 |
| 2f. Schema.devectorize() 8 fields | 1.24 | 1.20 | 1.30 | 1.40 | 1.20 | 3.30 | 809,651 |
| 2g. BudgetTracker.record_consumption() | 1.48 | 1.40 | 2.30 | 2.70 | 1.20 | 32.30 | 673,537 |
| 2h. BudgetTracker.rollback() | 0.45 | 0.40 | 0.70 | 0.80 | 0.40 | 2.70 | 2,223,210 |
| 2i. AuditChain.append() (growing chain) | 14.34 | 13.20 | 15.60 | 21.50 | 12.70 | 551.80 | 69,720 |
| 2j. AuditChain.verify() 100 records | 643.79 | 637.80 | 680.70 | 774.60 | 632.80 | 849.30 | 1,553 |
| 2k. AuditChain.verify() 500 records | 3210.36 | 3163.00 | 3578.00 | 4514.90 | 3114.80 | 4514.90 | 311 |

### 3. Enforcement Modes

| Benchmark | mean µs | median µs | p95 µs | p99 µs | min µs | max µs | ops/sec |
|-----------|--------:|----------:|-------:|-------:|-------:|-------:|--------:|
| 3a. default mode — APPROVE (feasible input) | 26.46 | 26.10 | 27.80 | 34.80 | 25.50 | 65.00 | 37,800 |
| 3b. default mode — PROJECT (infeasible input) | 634.69 | 622.50 | 688.40 | 969.60 | 611.50 | 1048.50 | 1,576 |
| 3c. reject mode — REJECT (infeasible input) | 22.58 | 21.40 | 26.20 | 49.40 | 20.80 | 127.70 | 44,288 |
| 3d. project mode — APPROVE (feasible input) | 24.99 | 24.20 | 26.20 | 46.20 | 23.70 | 112.10 | 40,020 |
| 3e. project mode — PROJECT (infeasible input) | 631.74 | 617.90 | 691.00 | 966.10 | 605.10 | 1483.90 | 1,583 |
| 3f. hybrid mode — APPROVE (feasible input) | 29.11 | 24.50 | 39.60 | 50.50 | 23.80 | 63.50 | 34,352 |
| 3g. hybrid mode — PROJECT (near boundary) | 627.74 | 618.20 | 686.20 | 797.60 | 606.50 | 1043.20 | 1,593 |
| 3h. default mode — REJECT (far outside, hard wall) | 279.49 | 274.50 | 301.80 | 360.20 | 270.60 | 707.20 | 3,578 |
| 3i. enforce with schema (8 fields) — APPROVE | 26.98 | 26.50 | 28.40 | 39.30 | 25.80 | 53.10 | 37,071 |
| 3j. enforce with schema (8 fields) — PROJECT | 631.18 | 621.70 | 677.60 | 875.00 | 608.30 | 1111.40 | 1,584 |

### 4. Extension Layer Latency

| Benchmark | mean µs | median µs | p95 µs | p99 µs | min µs | max µs | ops/sec |
|-----------|--------:|----------:|-------:|-------:|-------:|-------:|--------:|
| 4a. BreakerStateMachine.update() — CLOSED (healthy) | 1.05 | 1.00 | 1.10 | 1.20 | 1.00 | 4.00 | 952,744 |
| 4b. BreakerStateMachine.update() — THROTTLED (high load) | 1.14 | 1.10 | 1.20 | 1.30 | 1.00 | 3.50 | 877,963 |
| 4c. synthesize_envelope() — CLOSED mode | 5.53 | 5.50 | 5.60 | 6.50 | 5.40 | 8.80 | 180,933 |
| 4d. synthesize_envelope() — THROTTLED mode | 5.55 | 5.50 | 5.60 | 8.00 | 5.30 | 14.80 | 180,336 |
| 4e. build_v5_policy_from_envelope() | 57.82 | 56.80 | 60.20 | 81.10 | 55.40 | 117.00 | 17,296 |
| 4f. PolicyParser.parse() — 4-field config | 3.95 | 3.90 | 4.00 | 4.20 | 3.80 | 5.90 | 253,088 |
| 4g. PolicyParser.parse() — 8-field config | 7.24 | 7.20 | 7.40 | 7.50 | 7.00 | 10.70 | 138,085 |
| 4h. governor.enforce_next_step() — CLOSED (healthy) | 133282.31 | 132927.90 | 137012.80 | 139430.20 | 130380.80 | 156160.20 | 8 |
| 4i. NumerailPolicyContract.from_v5_config() | 1203.70 | 1174.45 | 1359.90 | 1762.90 | 1149.40 | 2648.10 | 831 |
| 4j. contract.verify_digest() | 1181.90 | 1151.90 | 1337.40 | 1934.30 | 1129.80 | 2061.50 | 846 |
| 4k. lint_config() — 2-field policy | 2.85 | 2.80 | 2.90 | 3.00 | 2.70 | 4.90 | 351,198 |

### 5. Throughput

| Benchmark | mean µs | median µs | p95 µs | p99 µs | min µs | max µs | ops/sec |
|-----------|--------:|----------:|-------:|-------:|-------:|-------:|--------:|
| 5a. NumerailSystemLocal.enforce() — 2 fields (full stack) | 160.92 | 146.00 | 183.10 | 258.30 | 142.40 | 9150.10 | 6,214 |
| 5b. NumerailSystemLocal.enforce() — 8 fields (full stack) | 192.00 | 182.70 | 251.70 | 308.10 | 175.70 | 781.60 | 5,208 |
| 5c. enforce() standalone — 4D box (APPROVE) | 25.03 | 24.65 | 26.70 | 35.50 | 23.70 | 54.30 | 39,957 |
| 5d. enforce() standalone — 16D linear (PROJECT) | 471.01 | 459.50 | 514.50 | 807.50 | 449.60 | 943.10 | 2,123 |
| 5e. enforce() + AuditChain.append() combined | 54.57 | 52.50 | 61.60 | 92.00 | 49.60 | 666.70 | 18,324 |
| 5f. NumerailSystemLocal full cycle (enforce + result access) | 161.81 | 157.70 | 184.70 | 237.00 | 152.20 | 654.70 | 6,180 |

### 6. Scaling Curves

| Benchmark | mean µs | median µs | p95 µs | p99 µs | min µs | max µs | ops/sec |
|-----------|--------:|----------:|-------:|-------:|-------:|-------:|--------:|
| 6a. enforce — 4 constraints, 8D (PROJECT) | 73.96 | 71.40 | 85.50 | 126.90 | 69.50 | 135.80 | 13,520 |
| 6a. enforce — 8 constraints, 8D (PROJECT) | 80.70 | 78.50 | 91.40 | 145.40 | 76.60 | 181.80 | 12,391 |
| 6a. enforce — 16 constraints, 8D (PROJECT) | 94.43 | 91.80 | 107.40 | 158.20 | 89.70 | 221.10 | 10,590 |
| 6a. enforce — 32 constraints, 8D (PROJECT) | 298.47 | 290.10 | 342.10 | 467.00 | 285.70 | 535.00 | 3,350 |
| 6a. enforce — 64 constraints, 8D (PROJECT) | 231.36 | 224.50 | 269.30 | 346.40 | 220.30 | 514.00 | 4,322 |
| 6a. enforce — 128 constraints, 8D (PROJECT) | 252.85 | 245.90 | 290.80 | 379.20 | 241.30 | 471.30 | 3,955 |
| 6b. enforce — 8D 2D linear (PROJECT) | 138.41 | 134.90 | 153.90 | 228.00 | 131.90 | 321.70 | 7,225 |
| 6b. enforce — 8D 4D linear (PROJECT) | 188.12 | 181.90 | 218.80 | 308.70 | 179.00 | 555.70 | 5,316 |
| 6b. enforce — 8D 8D linear (PROJECT) | 642.23 | 624.80 | 758.20 | 895.20 | 611.90 | 1192.30 | 1,557 |
| 6b. enforce — 8D 16D linear (PROJECT) | 476.54 | 457.90 | 595.40 | 804.40 | 449.90 | 888.40 | 2,098 |
| 6b. enforce — 8D 30D linear (PROJECT) | 658.00 | 635.10 | 810.70 | 1118.40 | 622.80 | 1212.40 | 1,520 |
| 6b. enforce — 8D 64D linear (PROJECT) | 1411.88 | 1387.85 | 1543.00 | 1801.40 | 1363.60 | 2425.50 | 708 |
| 6c. AuditChain.verify() — 10 records | 65.18 | 64.50 | 68.00 | 81.30 | 63.70 | 117.30 | 15,342 |
| 6c. AuditChain.verify() — 50 records | 327.27 | 322.75 | 350.00 | 475.30 | 317.70 | 475.30 | 3,056 |
| 6c. AuditChain.verify() — 100 records | 646.11 | 640.30 | 694.80 | 728.80 | 636.80 | 728.80 | 1,548 |
| 6c. AuditChain.verify() — 250 records | 1636.50 | 1615.50 | 1712.90 | 1933.10 | 1597.90 | 1933.10 | 611 |
| 6c. AuditChain.verify() — 500 records | 3348.82 | 3262.45 | 3646.10 | 4371.10 | 3170.80 | 4371.10 | 299 |
| 6c. AuditChain.verify() — 1000 records | 6779.90 | 6630.90 | 7418.60 | 11005.80 | 6376.90 | 11005.80 | 147 |
| 6d. BudgetTracker.record_consumption() — 1 budgets | 1.37 | 1.30 | 1.50 | 3.00 | 1.20 | 12.70 | 732,172 |
| 6d. BudgetTracker.record_consumption() — 2 budgets | 1.95 | 1.90 | 2.00 | 2.40 | 1.80 | 6.90 | 514,086 |
| 6d. BudgetTracker.record_consumption() — 4 budgets | 3.14 | 3.10 | 3.20 | 5.10 | 3.00 | 9.00 | 318,878 |
| 6d. BudgetTracker.record_consumption() — 8 budgets | 5.54 | 5.50 | 5.70 | 6.20 | 5.30 | 7.80 | 180,558 |

## Scaling Analysis

### Constraint Count Scaling (8D, PROJECT path)

| Constraints | mean µs | p99 µs | Relative to 4-constraint |
|------------:|--------:|-------:|-------------------------:|
| 4 | 73.96 | 126.90 | 1.00× |
| 8 | 80.70 | 145.40 | 1.09× |
| 16 | 94.43 | 158.20 | 1.28× |
| 32 | 298.47 | 467.00 | 4.04× |
| 64 | 231.36 | 346.40 | 3.13× |
| 128 | 252.85 | 379.20 | 3.42× |

### Dimension Count Scaling (PROJECT path)

| Dimensions | mean µs | p99 µs | Relative to 2D |
|----------:|--------:|-------:|---------------:|
| 2 | 138.41 | 228.00 | 1.00× |
| 4 | 188.12 | 308.70 | 1.36× |
| 8 | 642.23 | 895.20 | 4.64× |
| 16 | 476.54 | 804.40 | 3.44× |
| 30 | 658.00 | 1118.40 | 4.75× |
| 64 | 1411.88 | 1801.40 | 10.20× |

### AuditChain.verify() Scaling

| Records | mean µs | p99 µs | µs per record |
|--------:|--------:|-------:|--------------:|
| 10 | 65.18 | 81.30 | 6.5182 |
| 50 | 327.27 | 475.30 | 6.5454 |
| 100 | 646.11 | 728.80 | 6.4611 |
| 250 | 1636.50 | 1933.10 | 6.5460 |
| 500 | 3348.82 | 4371.10 | 6.6976 |
| 1000 | 6779.90 | 11005.80 | 6.7799 |

---

*Generated by `packages/numerail/tests/benchmark_performance.py`.*