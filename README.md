# heimdall-eval-harness

**A benchmark and scoring harness for one question: does an AI skill-tagger know when *not* to answer?**

[Heimdall](https://joinheimdall.com), a job-search product I built, reads job postings and tags which skills each requirement asks for, using a fixed list of 40 skills. The costly mistake is a **false tag**. When "Strong Python" gets tagged as *CI/CD*, the product tells a user they're missing a skill the job never asked for. So "decline to answer" has to be measured as carefully as accuracy.

This repository contains that measurement: a labelled benchmark, a scorer, baselines and a regression check. It doesn't include Heimdall's product code, prompts or user data. The live-model option below uses a simple prompt written for this repository.

## How it scores

For each requirement, a predictor returns one skill or *abstains*. Every case gets exactly one outcome:

| Outcome | Meaning |
|---|---|
| correct map | tagged the right skill |
| wrong map | tagged a different skill |
| missed map | abstained, but a skill fit |
| correct abstain | abstained, and nothing fit |
| **false positive** | **tagged a skill where nothing fit** (the costly error) |

The scorer reports mapping precision, recall and F1; abstention precision and recall; and false positives broken down by *why* the case should have been declined.

## The benchmark (42 cases)

**27 cases where a skill fits.** Four of them are paraphrases written to share almost no words with the skill definition, so word matching can't find them.

**15 cases where the right answer is to abstain**, in four categories:

| Category | Example | Why it should abstain |
|---|---|---|
| Technology name only | "Strong Python" | The list has no "knows tool X" skill. |
| Proficiency or credentials | "7+ years of professional software engineering experience" | No specific skill is named. |
| Gap in the taxonomy | "taking models from notebook to live traffic" | A real requirement the list doesn't cover. |
| Near miss | "familiarity with large language models" | On topic, but outside the skill's definition. |

Four cases are quoted from a public job posting. The rest were written for this benchmark and are labelled `"source": "constructed"`.

## Results

| Predictor | Accuracy | Mapping F1 | Abstention recall | False positives (of 15) | Missed fits | Cost for 42 cases |
|---|---|---|---|---|---|---|
| Always abstain (floor) | 0.36 | 0.00 | 1.00 | 0 | 27 | free |
| Word overlap (naive) | 0.79 | 0.80 | 0.60 | 6 | 0 | free |
| Word overlap + name-only rule | 0.83 | 0.82 | 0.80 | 3 | 1 | free |
| **gpt-4o-mini** | **0.98** | **0.98** | **0.93** | **1** | **0** | **$0.008** |
| gpt-4o | 0.98 | 0.98 | 0.93 | 1 | 0 | $0.140 |

The offline rows are reproducible with `npm run eval` and no API key. The model rows come from a single live run on 2026-11-02, with temperature 0 and one API call per case. The raw outputs, including each model's stated reason, are in [`eval/demand-mapping/results/`](eval/demand-mapping/results/).

### What the results show

1. **Both models clear the bar the rules couldn't.** Each scored 41 of 42, got every genuine fit right, and declined 14 of the 15 must-abstain cases. That includes the taxonomy gaps and near misses that no word rule could catch.
2. **gpt-4o costs 17× more for exactly the same score.** On this benchmark, the cheaper model is the right choice.
3. **Both models fail on the same case: "track record of taking models from notebook to live traffic."** The taxonomy has no skill for deploying machine-learning models, so the right answer is to decline. Instead, gpt-4o-mini tagged it as *CI/CD* and gpt-4o tagged it as *AI-Assisted Development*. Each model's stated reason explains the requirement correctly and then picks the closest-sounding skill anyway: the "topically close" failure this benchmark was built to catch. An earlier version of Heimdall's prompt failed on this same requirement.
4. **The benchmark is now too easy to tell the two models apart.** They differ on only one case, and both get it wrong. The next step is harder must-abstain cases, more requirements quoted from real postings, and repeated runs to measure consistency.

### What the offline baselines show

1. **Word overlap behaves exactly as feared.** It tagged 6 of the 15 must-abstain cases. "familiarity with large language models" and even "LLMs" both became *Prompt/System Design*, and "deploy services on AWS" became *Authentication*.
2. **A simple rule halves the false positives, but it hits a ceiling.** The rule says to abstain when a requirement is only a technology name plus proficiency words. It fixes 3 of the 6 false positives. The other 3 are 2 taxonomy gaps and 1 near miss, which need an understanding of meaning. No word rule can catch them.
3. **The rule has a real cost.** It also declines "evals", a genuine fit for *Evaluation*, because a rule can't tell a *practice* name from a *tool* name. The scorer reports this as a missed map instead of hiding it inside the overall accuracy.
4. **Paraphrases defeat word matching.** Word overlap mapped 3 of the 4 paraphrase cases to the wrong skill. An earlier version of the benchmark had no paraphrases, and word matching looked artificially good on it.

## The regression check

`baseline.json` holds the scores of the best offline predictor. CI reruns the benchmark on every push and fails if:

- any metric gets worse,
- **any single case** that used to pass now fails, even if the totals are unchanged, or
- **any new false positive** appears.

The tests also confirm that the committed baseline is reproduced exactly by an actual run, so it can't have been typed in by hand.

## Run it

Requires Node 18 or later. No dependencies to install.

```bash
npm test        # 15 tests: scorer maths on hand-checked inputs, benchmark validity, baseline reproduction, stubbed API parsing
npm run eval    # all offline predictors, comparison table, regression check (exit code 1 on regression)
```

Optional, with your own OpenAI key (42 calls per model; the run above cost about $0.15 in total):

```bash
OPENAI_API_KEY=... npm run eval:live    # gpt-4o-mini and gpt-4o
```

Live runs save every prediction, the model's stated reason, token counts and estimated cost to `eval/demand-mapping/results/`. Nothing in this repository is a made-up model output.

## Files

```
eval/demand-mapping/benchmark.json   the 42 labelled cases
eval/demand-mapping/scorer.mjs       outcome classification, metrics, regression check
eval/demand-mapping/run.mjs          offline predictors and command-line runner
eval/demand-mapping/live.mjs         optional OpenAI predictor (prompt, output schema, cost)
eval/demand-mapping/baseline.json    committed scores the regression check compares against
eval/demand-mapping/results/         raw outputs from the live gpt-4o-mini and gpt-4o runs
data/capabilities.json               the 40 skills with one-sentence definitions
tests/demand-mapping-eval.test.mjs   test suite (no network)
```

## Limitations

- **The labels were drafted with AI assistance** (Claude Code) and checked against the definitions. They need independent human review before anyone treats them as ground truth.
- **42 cases is enough to catch regressions** on known failure modes, but too few to estimate real-world accuracy.
- **Each requirement is scored on its own.** A full job posting gives more context, and that context can change the right answer.

Built with AI assistance (Claude Code).
