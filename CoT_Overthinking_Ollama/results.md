# Ollama CoT-Overthinking Results

Behavioural replication of "Probing the Trajectories of Reasoning Traces in LLMs" (arXiv:2601.23163) using `gemma3:4b` served by Ollama. No model internals used — answers are elicited by sampling, with a follow-up logit-based validation.

- Model: `gemma3:4b` (served by Ollama) / `google/gemma-3-4b-it` (tokenizer, and local transformers for logit validation)
- Dataset: `TIGER-Lab/MMLU-Pro`, `test` split, **math** category (1000 questions)
- Probe: forced continuation via `/api/chat` with `[system, user, assistant: "<think>{prefix}</think>\n"]`
- Decile points: `[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` (token-based cuts)
- `PROBE_TEMPERATURE = 0.0`, `PROBE_TOP_P = 1.0`, `PROBE_MAX_COMPLETION_TOKENS = 32`
- `probe_method_version = 3`

## 1. Baseline

- 1000 math questions baselined with Ollama / `gemma3:4b`.
- Baseline accuracy: **502/1001 (50.2%)** (one extra business-category row from an earlier smoke test).

## 2. Probe fidelity

Validated before launching the full run (20 questions):

| Decile | UNKNOWN | Reading |
|--------|---------|---------|
| 0.0 | 18/20 | Model still reasoning — no commitment |
| 0.3 | 11/20 | ~half starting to commit |
| 0.5 | 4/20 | Most have committed |
| 0.9 | 3/20 | Nearly all committed |
| 1.0 | 0/20 | Everyone commits |

- **Decile-1 match with baseline**: 19/20 (1 edge case attributable to tokenisation drift between `/api/generate` and `/api/chat`).
- Steady-state per-question cost: **~4.5 s** → ~76 min for 1000 questions × 1 condition.

## 3. Full resample run — accuracy by decile

Three conditions × 1000 math questions × 11 deciles = 33000 probes.

### `original`
| decile | UNK % | correct % |
|--------|-------|-----------|
| 0.0 | 95.6 | 0.4 |
| 0.1 | 86.1 | 1.8 |
| 0.2 | 69.4 | 5.5 |
| 0.3 | 57.9 | 9.6 |
| 0.4 | 49.9 | 11.5 |
| 0.5 | 41.1 | 14.0 |
| 0.6 | 35.8 | 20.6 |
| 0.7 | 32.1 | 23.7 |
| 0.8 | 26.1 | 27.4 |
| 0.9 | 22.9 | 35.0 |
| **1.0** | **9.0** | **51.9** |

### `shuffle` (tokens permuted — preserves length & token distribution, destroys ordering)
| decile | UNK % | correct % |
|--------|-------|-----------|
| 0.0 | 95.6 | 0.5 |
| 0.1 | 37.4 | 11.3 |
| 0.5 | 10.1 | 16.2 |
| 1.0 | 9.0 | 17.3 |

### `random` (uniform random tokens, excluding specials)
| decile | UNK % | correct % |
|--------|-------|-----------|
| 0.0 | 95.8 | 0.4 |
| 0.1 | 52.6 | 8.3 |
| 0.5 | 49.6 | 9.8 |
| 1.0 | 38.0 | 10.0 |

### Headline comparison at decile 1.0
| condition | correct % |
|-----------|-----------|
| original  | **51.9**  |
| shuffle   | 17.3      |
| random    | 10.0      |

The gap between `original` (51.9%) and `shuffle` (17.3%) is the "gains come from content, not length" effect: the same tokens in a scrambled order give a third of the accuracy. `random` is at chance (1 / 10 options).

## 4. UNKNOWN counts

Probes whose parsed answer was `UNKNOWN` (model did not commit a letter within `PROBE_MAX_COMPLETION_TOKENS = 32`):

| condition | UNKNOWN | % of 11,000 probes |
|-----------|---------|--------------------|
| original  | 5259    | 47.8% |
| shuffle   | 2351    | 21.4% |
| random    | 5644    | 51.3% |

Heavily skewed to low deciles (~956/1000 at decile 0 alone). At decile ≥ 0.5 the `original` UNKNOWN rate drops to ~29%.

## 5. Logit validation of UNKNOWN probes

To check whether UNKNOWN represents genuine model non-commitment or lost data, a stratified sample of UNKNOWN probes was re-scored locally using `gemma-3-4b-it` via transformers. For each sampled probe we reconstruct the exact Ollama chat-template prompt, forward-pass, take the next-token logits, and measure the probability mass on the ten answer-letter tokens A–J.

- Sample: 200 UNKNOWN probes per condition, stratified across deciles.
- Threshold for "lost data": letter-restricted argmax is the correct answer AND total letter mass ≥ 0.1 at the next-token position.

| metric | original | shuffle | random | all |
|--------|----------|---------|--------|-----|
| UNKNOWN justified | 97.0% | 97.0% | 94.5% | **96.2%** |
| "lost data" (confident & correct) | 3.0% | 3.0% | 5.5% | 3.8% |
| mean letter-mass at next token | 0.12 | 0.26 | 0.47 | 0.28 |
| argmax letter = correct answer | — | — | — | 11.8% |

Interpretation:

- **96.2%** of UNKNOWNs are justified: the model either wanted to continue reasoning (low letter mass) or its letter-argmax was an incorrect answer.
- **3.8%** represent genuine lost data (model was in commit mode AND would have picked the correct answer).
- Argmax-correct rate of 11.8% is barely above the 10% chance baseline, confirming UNKNOWN does not hide correct answers.
- `random` has the highest letter mass (0.47) — when fed gibberish the model stops reasoning and commits, but its commit is near-random, so still "justified".

**Conclusion: treating UNKNOWN as incorrect in downstream analysis is defensible.**

## 6. Right-then-wrong transitions (the main behavioural question)

Questions in the `original` condition where the probe answer was correct at some decile < 1.0 but the final answer (decile 1.0) was wrong:

| outcome | count / 1000 |
|---------|--------------|
| Right-and-stayed-right | 396 |
| **Right-then-wrong** | **108** |
| Wrong-then-right | 123 |
| Final correct (total) | 519 |
| Final wrong (total) | 481 |

- 108 / 481 ≈ **22.5%** of final-wrong answers had the correct answer earlier in the CoT.
- This is a lower bound: UNKNOWN probes are not counted as "correct-at-this-decile". Given only ~3.8% of UNKNOWNs had the model confidently correct, the true count is probably ~108 + a few dozen.

## 7. Files

| path | contents |
|------|----------|
| `baseline/baseline_CoTs_options_ollama.jsonl` | 1000 math baseline CoTs |
| `resample_results/options_results_ollama.jsonl` | `original` condition (1000 rows × 11 deciles) |
| `resample_results/options_shuffle_results_ollama.jsonl` | `shuffle` condition |
| `resample_results/options_random_results_ollama.jsonl` | `random` condition |
| `resample_results/unknown_logit_validation.jsonl` | 600 UNKNOWN probes re-scored via logits |
| `ollama_options.py` | Baseline CoT generator (Ollama) |
| `ollama_option_resample.py` | Decile-truncation probe (Ollama) |
| `unknown_logit_validation.py` | Logit-based UNKNOWN validator (transformers) |
