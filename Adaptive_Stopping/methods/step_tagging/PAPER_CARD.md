# Paper implementation card: Step-Tagging Early-Stopping (ST-ES)

**Full title:** *Step-Tagging Early-Stopping: Toward controlling the generation of Language Reasoning Models through black-box step monitoring*

- **Paper / OpenReview:**
  - https://openreview.net/forum?id=bNn5OusO22 (AdaptFM @ ICML 2026, poster session 1). The title was checked against the official accepted-paper list.
  - arXiv [2512.14332](https://arxiv.org/abs/2512.14332), v2 of 16 Jul 2026.
  - The team also has the 22-page proceedings-format PDF (`75_Step_Tagging_Early_Stopping.pdf`). We compared its text with arXiv v2 line by line. It is identical except for Algorithm 1, which adds a *minimal step size* k ("Generate step s_i … where |s_i| > k"), and no value for k is given. Eq. 1, Eq. 3, the 100-token answer budget and all reported numbers are the same in both.
  - Authors: Y. Belkhiter, S. Tirupathi, G. Zizzo, J. D. Kelleher.
- **Primary module:** `Adaptive_Stopping/`
- **Secondary modules:** `Backbones/`, `Dataloaders/`, `Performance_Metrics/`
- **Owner / reviewer:** team of Devansh Jangid (@devanshjangid2005), Sanidhya Srivastava, Arnav Tiku, Aman Gupta, Karan Malik. Tracking issue: krsnoki/DA4022#24.
- **Status:** `partial-reproduction`
- **Upstream code / licence:**
  - No code release for ST-ES was found (paper, arXiv page, GitHub search, 2026-11-25).
  - The first author's MIT-licensed [TRACES](https://github.com/yannisbelkhiter/TRACES) repository belongs to a later paper whose stopping rule (a constructive/evaluative ratio with a sliding window) is different. It was **not** implemented. It was consulted only to confirm details the ST-ES paper also states: the `"\n\n"` split, the 14 annotation label names, and the prompt-guided baseline prompts.
  - The 99 GPT-4o-mini-labelled validation steps come from the authors' MIT-licensed [ReasonType_Human_validation](https://github.com/yannisbelkhiter/ReasonType_Human_validation) (commit `efb7676`). They are downloaded at run time and not committed.
- **Backbone IDs:** `text.deepseek-r1-distill-qwen-1.5b` (revision `ad9f0ae`)
- **Dataset IDs and split:**
  - GSM8K: calibration on `text.gsm8k.train` (200 problems), evaluation on `text.gsm8k.test` (200).
  - MATH: calibration on `text.math.train` (150), evaluation on `text.math500.test` (150).
  - All subsets use seed 42.
- **Hardware assumptions:**
  - One CUDA GPU with at least 8 GB (tested: RTX 4060 Laptop GPU, 8 GB, Windows 11, driver 610.88).
  - The policy itself is CPU-only.
  - Tests run on CPU.

## Research summary

### Problem
Reasoning models over-generate. After they have effectively found the answer, they keep verifying, re-checking and reflecting, which costs tokens, latency and energy. Existing dynamic early-exit criteria (confidence or entropy) need access to model internals (white-box or grey-box). Static budgets ignore what the model is actually doing.

### Method
1. **Segment the generated reasoning into steps** at every `"\n\n"` (Eq. 1).
2. **Tag each step** with a reasoning type from *ReasonType*, a 14-class taxonomy (problem re-statement, formula substitution, verification, self-talk, exploration, ...). The paper derived the taxonomy by having GPT-4o-mini label steps. It labels steps with GPT-4o-mini and trains one binary `bert-base-uncased` classifier per step type on those labels.
3. **Stop rule (Eq. 3, Alg. 1):** for a chosen step type τ\* and threshold δ, continue while the running count of τ\* steps is ≤ δ. Stop at the first step that makes the count exceed δ.
4. **Answer forcing:** after stopping, prompt the model for its current best answer, with 100 extra tokens.
5. **Calibration:** sweep τ\* over the step types and δ ∈ [0, 20] on a training split. Keep the configurations that preserve 85%, 90% and 95% of standard accuracy. Apply them to the held-out test split.

The whole criterion is black-box: it reads only the generated text.

### Expected behaviour
- **Token savings:** the paper reports 20–50% fewer tokens than standard generation, with accuracy between +2% and −12% depending on the target.
- **Comparison with the token-count baseline:** on the training split, ST-ES curves for late-reasoning types (verification, self-talk) match or beat the token-count (step-count) baseline.
- **Latency and memory:** the paper reports token counts, not measured latency. Its runtime analysis (App. F) is an estimate that assumes runtime is linear in tokens.

Paper-reported test results (Table 6, mean of 5 seeds; "Avg@5" = mean accuracy):

| Model | Dataset | Config | Tokens | Saved | Avg@5 |
|---|---|---|---:|---:|---:|
| DS-R1-Distill-Llama-8B | GSM8K | standard | 958.3 | – | 0.829 |
| | | ST-ES 85% / 90% / 95% | 521.5 / 618.8 / 696.7 | 45.6 / 35.4 / 27.3% | 0.708 / 0.788 / 0.809 |
| DS-R1-Distill-Llama-8B | MATH500 | standard | 3655.0 | – | 0.878 |
| | | ST-ES 85% / 90% / 95% | 1801.4 / 2120.3 / 2614.5 | 50.7 / 42.0 / 28.5% | 0.761 / 0.793 / 0.845 |
| DS-R1-Distill-Qwen-14B | GSM8K | standard | 662.9 | – | 0.910 |
| | | ST-ES 85% / 90% / 95% | 377.2 / 415.9 / 464.6 | 43.1 / 37.3 / 29.9% | 0.527 / 0.621 / 0.865 |
| DS-R1-Distill-Qwen-14B | MATH500 | standard | 3388.8 | – | 0.923 |
| | | ST-ES 85% / 90% / 95% | 2158.4 / 2393.6 / 2851.9 | 36.3 / 29.4 / 15.8% | 0.827 / 0.853 / 0.909 |

## Implementation

- **Input contract:**
  - The policy (`StepTagEarlyStopping.should_stop(state, metrics, budget)`) receives a `StepState` for every completed step: step index, step text (`str`, or `None` for missing telemetry), running generated-token count, and whether the model is still inside `<think>`.
  - Optional `metrics={"tag": ...}` passes a precomputed tag.
  - Optional `budget={"max_steps", "max_tokens"}` adds hard caps; a zero budget stops at the first step.
  - The live loop runs at batch size 1 inside `model.generate` as a stopping criterion. The decision is made at the token that closes `"\n\n"`.
- **Output contract:** `generate_with_policy(...)` returns:
  - the generated token IDs, whether the policy stopped, the stop step and the cut (number of kept tokens);
  - the forced-answer token IDs;
  - timings (end-to-end, TTFT, reasoning, answer, policy time);
  - the policy report: tag counts, count of τ\*, stop reason.
  - `answer_text()` gives the scored text.
- **Configuration:**
  - [`configs/st_es/gsm8k.yaml`](../../configs/st_es/gsm8k.yaml) and [`configs/st_es/math500.yaml`](../../configs/st_es/math500.yaml).
  - Key settings: seed 42; sampling T=0.6, top-p 0.95, top-k 50; bf16; token cap 4096 (GSM8K) / 8192 (MATH).
  - Tagger `keyword_reasontype.v1`; exit prompt `exit_boxed.v1` = `</think>\n\n**Final Answer**\n\boxed{` followed by ≤100 greedy tokens.
  - Grid: 13 step types × δ ∈ [0, 20], plus step counts {1…10, 12, …, 500}. Targets 85/90/95%.
  - Every run writes a manifest with model revision, counted parameters, GPU, library versions and the trace sha256.
- **Reproduction command:**
  ```bash
  python -m Adaptive_Stopping.experiments.st_es.run_all --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
  python -m Adaptive_Stopping.experiments.st_es.validate_tagger
  python -m Adaptive_Stopping.experiments.st_es.make_report --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
  ```
- **Tests:**
  - Command: `python -m pytest Adaptive_Stopping/tests Dataloaders/tests -q`
  - Count: 120+ tests, all on CPU, nothing downloaded.
  - They cover segmentation versus `str.split` under arbitrary chunking, tagger cases, δ/budget edge cases, and fast-sweep versus policy replay equivalence.
  - A tiny random Qwen2 test checks that batched continuation generation equals plain greedy decoding, that shared-prefix forced answers equal fresh prefill, and that the live stop equals the replay stop.

## Results

_Runs in progress (GSM8K and MATH-500 replay, plus the live latency and memory benchmark). Result tables will be added here and in `Performance_Metrics/results/st_es.csv`._

## Differences and limitations

- **Backbone.** We use DeepSeek-R1-Distill-Qwen-1.5B, not the 8B/14B/32B models. A 1.5B model reasons differently (more repetition, more formula steps), so the absolute numbers are not comparable with the paper's.
- **Step tagger (main deviation, `approximate`).** We use a rule-based keyword tagger instead of GPT-4o-mini labels and binary BERT taggers, because there was no API budget or labelled data. On 99 GPT-4o-mini-labelled steps released by the authors (DS-14B, MATH500, not used to write the rules), it agrees on 40.4% of 14-way labels. One-vs-rest macro-F1 is 0.48–0.88 depending on step type (verification 0.63, self-talk 0.69, exploration 0.88) (the paper's BERT taggers score about 0.8–0.9 on their own data). The step types it finds are therefore only an approximation of the paper's.
- **Seeds and subsets.** One seed (42) instead of five, so we report pass@1, not Avg@5. Subsets have 150–200 problems; GPQA, MMLU-Pro and AIME were not run.
- **Token cap.** 4096 (GSM8K) and 8192 (MATH) generated tokens instead of 32768. The same cap applies to the baseline and to ST-ES. Traces that hit the cap count as wrong unless an early stop rescues them.
- **Prompt.** The paper's standard prompt is the raw question (as in the authors' code). We append the DeepSeek model card's "reason step by step … \boxed{}" directive, which helps a 1.5B model produce a parseable answer.
- **Exit prompt.** The paper does not give its exit prompt. Ours closes the thinking block and opens `\boxed{`. The answer is decoded greedily and stops when the box closes (≤100 tokens).
- **Monitoring scope.** Steps after `</think>` (the final-answer section) are never used to stop. The paper does not discuss this case.
- **Minimal step size.** Steps are split at every `"\n\n"`, as in Eq. 1, arXiv v2 and the authors' code (k = 0). The proceedings version's minimal step size k (Algorithm 1) has no value in the paper, so it is not applied in the main results.
- **Baselines.**
  - The paper's "token-count baseline" stops after k steps of any type (`StepCountStopping`). It is calibrated with the same 85/90/95% rule.
  - The prompt-guided baselines P(0)user, P(0)system, P(1)system, P(3)system were **not run** for lack of time.
- **Offline replay versus live run.** Like the paper, accuracy and token counts come from offline replay over cached generations. Latency and peak memory come from one live run of the final configuration on a small subset (batch size 1). Replay-derived runtimes, where shown, are labelled *estimated*.
- **Cached-generation batching.** Traces are sampled in continuation rounds: per-row prefill, rounds of 1024 tokens, re-queued until EOS or the cap. This changes the random stream, not the sampling distribution. The GSM8K calibration traces were produced in two sessions because the first was interrupted; the manifest records how many records were reused.
- **Correctness checking.** Math-Verify runs without timeouts on Windows, where its subprocess-based timeout fails.
- **Not measured.** Energy (`not_measured`).

## Reproduction checklist

- [x] Exact paper link and version verified. The official AdaptFM accepted-papers page (fetched 2026-11-25) links `openreview.net/forum?id=bNn5OusO22` to this exact title and author list, and arXiv v2 matches it. The OpenReview page itself timed out from our network, so a camera-ready PDF or code link posted only there is unchecked.
- [x] Base model and dataset revisions pinned (registry entries)
- [x] Baseline runs on the same setup (standard and step-count, same traces, prompts and cap)
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported (replay quality and tokens; live latency and peak memory)
- [x] Licence/data terms checked (model MIT; GSM8K and MATH MIT; MATH-500 card states no licence and is recorded as `unknown`)
