# Steering experiment: feature 4103 at layer 17

Model `google/gemma-3-4b-it`, SAE `gemma-scope-2-4b-it-res/layer_17_width_16k_l0_medium`, greedy decode, max_new_tokens=2048.

Intervention: residual stream at layer 17 output += α · unit(W_dec[4103]) at every token during prompt + generation.

Legend: `X✓` = parsed answer X, matches actual; `X✗` = parsed X, wrong; `None✗` = failed to parse a letter.

## Set A — right->wrong (ollama) + feature fires at prompt
*Hypothesis: suppression (α < 0) should flip some wrong → correct.*

### Set A: 12 questions

| qid | actual | -500 | -200 | +0 | +200 | +500 |
|---|---|---|---|---|---|---|
| 8002 | F | F✓ | F✓ | H✗ | E✗ | F✓ |
| 8110 | C | A✗ | A✗ | A✗ | A✗ | C✓ |
| 8322 | A | D✗ | D✗ | D✗ | D✗ | D✗ |
| 7690 | A | H✗ | F✗ | F✗ | H✗ | A✓ |
| 8123 | I | I✓ | A✗ | None✗ | I✓ | I✓ |
| 7802 | A | J✗ | A✓ | J✗ | A✓ | A✓ |
| 7882 | A | None✗ | None✗ | None✗ | None✗ | None✗ |
| 8017 | E | None✗ | E✓ | E✓ | E✓ | E✓ |
| 7796 | I | I✓ | I✓ | C✗ | I✓ | I✓ |
| 8440 | I | I✓ | A✗ | D✗ | A✗ | F✗ |
| 8006 | A | B✗ | A✓ | A✓ | G✗ | None✗ |
| 8485 | E | D✗ | D✗ | D✗ | D✗ | D✗ |

**Flips vs +0 baseline (within-experiment):**

| condition | same | correct→wrong | wrong→correct | wrong→different-wrong |
|---|---|---|---|---|
| -500 | 5 | 2 | 4 | 1 |
| -200 | 7 | 0 | 3 | 2 |
| +0 | (baseline) | - | - | - |
| +200 | 5 | 1 | 3 | 3 |
| +500 | 4 | 1 | 6 | 1 |

**Accuracy per condition:**

| condition | correct / n | % |
|---|---|---|
| -500 | 4/12 | 33 |
| -200 | 5/12 | 42 |
| +0 | 2/12 | 17 |
| +200 | 4/12 | 33 |
| +500 | 7/12 | 58 |

## Set B — right->right + feature fires (selectivity control)
*If suppression helps selectively on A, it should not ruin B.*

### Set B: 2 questions

| qid | actual | -500 | -200 | +0 | +200 | +500 |
|---|---|---|---|---|---|---|
| 8021 | B | B✓ | None✗ | B✓ | B✓ | B✓ |
| 7781 | B | D✗ | B✓ | B✓ | D✗ | C✗ |

**Flips vs +0 baseline (within-experiment):**

| condition | same | correct→wrong | wrong→correct | wrong→different-wrong |
|---|---|---|---|---|
| -500 | 1 | 1 | 0 | 0 |
| -200 | 1 | 1 | 0 | 0 |
| +0 | (baseline) | - | - | - |
| +200 | 1 | 1 | 0 | 0 |
| +500 | 1 | 1 | 0 | 0 |

**Accuracy per condition:**

| condition | correct / n | % |
|---|---|---|
| -500 | 1/2 | 50 |
| -200 | 1/2 | 50 |
| +0 | 2/2 | 100 |
| +200 | 1/2 | 50 |
| +500 | 1/2 | 50 |

## Set C — right->right + feature silent
*Hypothesis: induction (α > 0) should flip some correct → wrong.*

### Set C: 12 questions

| qid | actual | -500 | -200 | +0 | +200 | +500 |
|---|---|---|---|---|---|---|
| 8490 | A | A✓ | A✓ | A✓ | A✓ | A✓ |
| 8282 | B | B✓ | B✓ | B✓ | B✓ | B✓ |
| 8401 | A | A✓ | A✓ | A✓ | A✓ | None✗ |
| 8101 | J | J✓ | J✓ | J✓ | J✓ | J✓ |
| 7664 | C | C✓ | C✓ | C✓ | C✓ | C✓ |
| 7749 | A | A✓ | A✓ | A✓ | A✓ | A✓ |
| 8480 | C | J✗ | None✗ | C✓ | C✓ | C✓ |
| 8374 | C | C✓ | C✓ | C✓ | C✓ | C✓ |
| 7726 | A | A✓ | A✓ | A✓ | None✗ | A✓ |
| 8502 | I | G✗ | A✗ | G✗ | G✗ | J✗ |
| 7668 | D | D✓ | D✓ | D✓ | D✓ | B✗ |
| 7893 | B | B✓ | E✗ | B✓ | B✓ | C✗ |

**Flips vs +0 baseline (within-experiment):**

| condition | same | correct→wrong | wrong→correct | wrong→different-wrong |
|---|---|---|---|---|
| -500 | 11 | 1 | 0 | 0 |
| -200 | 9 | 2 | 0 | 1 |
| +0 | (baseline) | - | - | - |
| +200 | 11 | 1 | 0 | 0 |
| +500 | 8 | 3 | 0 | 1 |

**Accuracy per condition:**

| condition | correct / n | % |
|---|---|---|
| -500 | 10/12 | 83 |
| -200 | 9/12 | 75 |
| +0 | 11/12 | 92 |
| +200 | 10/12 | 83 |
| +500 | 8/12 | 67 |

