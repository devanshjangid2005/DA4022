# D: Reproduction check and review

Part of the ST-ES implementation ([paper card](../PAPER_CARD.md)). This page records what was checked, how, and what was found.

## 1. Fresh-clone check (CPU)

On 2026-11-26 we cloned the branch from the public fork into an empty directory and ran the full test suite there. This catches dependence on files that exist only in the development checkout.

```bash
git clone --branch st-es-step-tagging --single-branch https://github.com/devanshjangid2005/DA4022.git DA4022_fresh
cd DA4022_fresh
python -m pytest Adaptive_Stopping/tests Dataloaders/tests Performance_Metrics/tests -q
```

| Item | Value |
|---|---|
| Commit | `6d37f36` |
| Result | **176 passed** in 8 s (CPU only, `CUDA_VISIBLE_DEVICES=""`). After the fixes below, the suite has 180 tests. |
| Platform | Windows 11 10.0.26200, Python 3.12.14, torch 2.11.0+cu128, transformers 5.24.1 |

## 2. Integration check

`tools/make_bundles.py` (outside the repository, next to it) rebuilds the integrated tree from upstream:

1. Start from a clean worktree of upstream `krsnoki/DA4022@main` (`368db37`).
2. Apply the core patch.
3. Apply the four workstream patches A–D. Each workstream only adds files, so they apply in any order.
4. Check that all 83 changed files are byte-identical to the branch, and run the test suite.

The latest report, with the exact commits and test count, is in `team_deliverables/INTEGRATION_CHECK.txt`. Every run so far has passed: all patches apply cleanly, no file differs from the branch, and the full test suite passes.

## 3. End-to-end smoke run (GPU)

`configs/st_es/gsm8k_smoke.yaml` uses 20 calibration and 20 test problems and 3 live requests, with otherwise the same settings as the main run. It exercises every stage (generation, calibration replay, test replay, live benchmark, report) in about 10 minutes.

```bash
python -m Adaptive_Stopping.experiments.st_es.run_all --config Adaptive_Stopping/configs/st_es/gsm8k_smoke.yaml --run-dir <tmp>
```

We ran it from the fresh clone with a separate run directory, in two attempts.

**First attempt (commit `6d37f36`): found a real bug.**
- With a non-default `--run-dir`, `run_all` passed the directory only to the generation stage. Calibration wrote its `selected.json` to the default run directory, and the evaluation stage could not find it.
- The main runs never hit this, because they use the default directory.
- Fixed in `a39cd1a`: `replay_sweep` and `live_benchmark` take `--run-dir` and `run_all` passes it to every stage. The new test `tests/test_run_all.py` checks the wiring for every stage and validates all configs.

**Second attempt (commit `a39cd1a`): all stages passed.** `run_all` exited 0, and `make_report` exited 0.

| Stage | Result |
|---|---|
| Generation, 20 calibration + 20 test problems | 149 s and 140 s; 273 and 198 tok/s, with small batches in the tail |
| Calibration replay (full grid; 736 forced answers in 21 s) | Standard 0.90 / 2030 tokens; selects formula substitution δ=5 (85%) and exploration δ=0 (90/95%) |
| Test replay | Standard 0.85 / 1387 tokens; ST-ES exploration δ=0 0.80 / 781 (−44%); step-count k=14 0.80 / 816 |
| Live, 3 requests | ST-ES stopped on 2 of 3, all answers correct, median latency 36.8 s → 23.4 s; every stop matched an offline replay of the same tokens |
| Wall time | About 10 minutes from scratch (about 6 minutes with cached traces) |

The numbers come from 20 problems and only show that the pipeline works; they are not results.

**Another hazard found while reviewing the report step.** `make_report` named its output folder and metrics rows after the evaluation dataset. A smoke run made inside the main repository would therefore have overwritten the main GSM8K results. Fixed in `e8eb50a`: configs can set `report_name` and `write_metrics: false`, and the smoke config does.

## 4. Paper version and links

- **Official AdaptFM accepted-paper list:** checked 2026-11-25. It links `openreview.net/forum?id=bNn5OusO22` to this exact title and author list.
- **Proceedings-format PDF:** compared line by line with arXiv v2. The only substantive difference is the minimal step size k in Algorithm 1 (see [B](B_min_step_size.md)).
- **OpenReview page itself:** **not verified.** The HTML page timed out from our network on 2026-11-25. On 2026-11-26 the OpenReview API answered with a bot challenge (`HTTP 403 ChallengeRequiredError`). Someone must open https://openreview.net/forum?id=bNn5OusO22 in a browser and record two things in issue #24: whether a camera-ready PDF newer than arXiv v2 is attached, and whether a code link is given.

## 5. Review notes

These are points a reviewer should know, in rough order of how much they affect the results.

1. **Approximate tagger.**
   - The rule-based tagger agrees with GPT-4o-mini on only 40% of 14-way labels, and verification recall is 3/9 (see [C](C_tagger_analysis.md)).
   - ST-ES results measure "ST-ES with this tagger". They are labelled `approximate`.
2. **Live and replay accuracy differ on GSM8K. This was diagnosed and is not a bug.**
   - In the 30-problem live run, ST-ES scored 0.70 against 0.90 for standard generation; the 200-problem replay shows only −1.5 points.
   - A re-run of the 6 failing problems showed that every live stop equals an offline replay of the same tokens, and that ST-ES follows standard's tokens exactly up to the stop.
   - The losses come from δ=0 stopping at the model's first check of an *intermediate* result. In the replay, these losses are offset by early stops that recover traces which hit the token cap (see the paper card).
3. **Sampling and batching.** Cached traces are sampled in batched continuation rounds. The distribution is unchanged, but the exact token streams depend on the batch schedule. Live runs sample at batch size 1, so live and replay traces of the same problem differ.
4. **Token caps.** 4096 for GSM8K and 8192 for MATH, against 32768 in the paper. 14% (GSM8K) and 17% (MATH-500) of standard test traces hit the cap and are scored as generated.
5. **Math-Verify on Windows.** Its timeouts are disabled because the subprocess-based timeout fails there. A pathological expression could hang the scorer; none did in our runs.
6. **transformers 5.x API.** The code uses `generate(cache=...)`, `StaticCache` and `DynamicCache.layers`, and pins `transformers==5.24.1`.
7. **GPU memory on Windows.** Exceeding dedicated memory spills into shared memory instead of raising OOM. The scripts cap the allocator at 85% and use static KV caches with chunked prefill. Budgets are tuned for 8 GB and can be raised on larger GPUs (`--kv-token-budget`).
8. **Report provenance.** `make_report.py` stamps the commit at report time. The traces themselves are identified by their sha256 in `provenance.json`.
