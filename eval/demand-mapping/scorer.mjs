// Evaluation harness for the capability-demand extractor -- scoring,
// false-positive analysis, and regression gating.
//
// WHAT IS MEASURED: for each benchmark case (one requirement phrase with
// an expected canonical capability, or null = the correct answer is to
// ABSTAIN), a predictor returns either a canonical capability name or
// null. Every case lands in exactly one outcome:
//
//   correct_map      expected X, predicted X
//   wrong_map        expected X, predicted Y (Y != X, Y != null)
//   missed_map       expected X, predicted null (over-abstention)
//   correct_abstain  expected null, predicted null
//   false_positive   expected null, predicted some capability
//
// false_positive is reported separately from ordinary errors on purpose:
// for this extractor a false positive becomes a skill tag in the product
// that tells a user they are "missing" something the job never asked
// for (e.g. "Strong Python" -> CI/CD). Over-abstention only loses signal.
//
// Pure functions, no I/O except loadBenchmark/loadCanonicalCatalog, which
// read files from this repository.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const OUTCOMES = ["correct_map", "wrong_map", "missed_map", "correct_abstain", "false_positive"];

export function classifyOutcome(expected, predicted) {
  const exp = expected ?? null;
  const pred = predicted ?? null;
  if (exp === null) return pred === null ? "correct_abstain" : "false_positive";
  if (pred === null) return "missed_map";
  return pred === exp ? "correct_map" : "wrong_map";
}

const ratio = (num, den) => (den === 0 ? null : num / den);
const round = (x) => (x === null ? null : Math.round(x * 10000) / 10000);
function f1(p, r) {
  if (p === null || r === null || p + r === 0) return p === 0 || r === 0 ? 0 : null;
  return (2 * p * r) / (p + r);
}

// cases: [{id, expected, category}]; predictions: {caseId: capability|null}
// A case with no prediction entry is scored as an abstention (null) and
// listed in `missingPredictions` so a crashed predictor is visible.
export function scorePredictions(cases, predictions) {
  const counts = Object.fromEntries(OUTCOMES.map((o) => [o, 0]));
  const byCategory = {};
  const perCase = [];
  const missingPredictions = [];

  for (const c of cases) {
    if (!(c.id in predictions)) missingPredictions.push(c.id);
    const predicted = predictions[c.id] ?? null;
    const outcome = classifyOutcome(c.expected, predicted);
    counts[outcome] += 1;
    const cat = (byCategory[c.category] ??= { total: 0, ...Object.fromEntries(OUTCOMES.map((o) => [o, 0])) });
    cat.total += 1;
    cat[outcome] += 1;
    perCase.push({ id: c.id, category: c.category, expected: c.expected ?? null, predicted, outcome });
  }

  const expectedMapped = cases.filter((c) => c.expected != null).length;
  const expectedAbstain = cases.length - expectedMapped;
  const predictedMapped = counts.correct_map + counts.wrong_map + counts.false_positive;
  const predictedAbstain = counts.correct_abstain + counts.missed_map;

  const mappingPrecision = ratio(counts.correct_map, predictedMapped);
  const mappingRecall = ratio(counts.correct_map, expectedMapped);
  const abstentionPrecision = ratio(counts.correct_abstain, predictedAbstain);
  const abstentionRecall = ratio(counts.correct_abstain, expectedAbstain);

  const falsePositivesByCategory = {};
  for (const [cat, v] of Object.entries(byCategory)) {
    const mustAbstain = cases.filter((c) => c.category === cat && c.expected == null).length;
    if (mustAbstain > 0) falsePositivesByCategory[cat] = { false_positives: v.false_positive, must_abstain_cases: mustAbstain, rate: round(v.false_positive / mustAbstain) };
  }

  return {
    total: cases.length,
    counts,
    metrics: {
      accuracy: round(ratio(counts.correct_map + counts.correct_abstain, cases.length)),
      mapping_precision: round(mappingPrecision),
      mapping_recall: round(mappingRecall),
      mapping_f1: round(f1(mappingPrecision, mappingRecall)),
      abstention_precision: round(abstentionPrecision),
      abstention_recall: round(abstentionRecall),
      false_positive_rate: round(ratio(counts.false_positive, expectedAbstain)),
    },
    false_positives_by_category: falsePositivesByCategory,
    by_category: byCategory,
    failures: perCase.filter((p) => p.outcome !== "correct_map" && p.outcome !== "correct_abstain"),
    per_case: perCase,
    missing_predictions: missingPredictions,
  };
}

// Metrics where HIGHER is better, and the one where LOWER is better.
const HIGHER_IS_BETTER = ["accuracy", "mapping_precision", "mapping_recall", "mapping_f1", "abstention_precision", "abstention_recall"];
const LOWER_IS_BETTER = ["false_positive_rate"];

// Regression gate: compares a new score against a committed baseline.
// A regression is (a) any metric moving in the bad direction by more than
// `tolerance`, or (b) any single case that was correct in the baseline and
// is wrong now -- aggregate metrics can hide a traded-off case, so
// per-case regressions are reported even when totals look flat. A new
// false positive is always a regression, regardless of tolerance.
export function compareToBaseline(current, baseline, { tolerance = 0 } = {}) {
  const regressions = [];
  const improvements = [];

  for (const m of [...HIGHER_IS_BETTER, ...LOWER_IS_BETTER]) {
    const now = current.metrics[m];
    const before = baseline.metrics[m];
    if (now === null || before === null || now === undefined || before === undefined) continue;
    const delta = round(now - before);
    const worse = LOWER_IS_BETTER.includes(m) ? delta > tolerance : delta < -tolerance;
    const better = LOWER_IS_BETTER.includes(m) ? delta < 0 : delta > 0;
    if (worse) regressions.push({ kind: "metric", metric: m, baseline: before, current: now, delta });
    else if (better) improvements.push({ kind: "metric", metric: m, baseline: before, current: now, delta });
  }

  const baselineById = new Map((baseline.per_case ?? []).map((p) => [p.id, p]));
  const isCorrect = (o) => o === "correct_map" || o === "correct_abstain";
  for (const p of current.per_case) {
    const before = baselineById.get(p.id);
    if (!before) continue;
    if (isCorrect(before.outcome) && !isCorrect(p.outcome)) {
      regressions.push({ kind: "case", id: p.id, baseline: before.outcome, current: p.outcome, predicted: p.predicted, expected: p.expected });
    } else if (!isCorrect(before.outcome) && isCorrect(p.outcome)) {
      improvements.push({ kind: "case", id: p.id, baseline: before.outcome, current: p.outcome });
    }
    if (p.outcome === "false_positive" && before.outcome !== "false_positive" && !regressions.some((r) => r.kind === "case" && r.id === p.id)) {
      regressions.push({ kind: "case", id: p.id, baseline: before.outcome, current: p.outcome, predicted: p.predicted, expected: p.expected });
    }
  }

  return { passed: regressions.length === 0, regressions, improvements };
}

// Side-by-side comparison of several predictors/models on the same
// benchmark, ordered by fewest false positives, then mapping F1.
export function comparePredictors(results) {
  return results
    .map(({ name, score, cost_usd = null, latency_ms = null }) => ({
      predictor: name,
      accuracy: score.metrics.accuracy,
      mapping_precision: score.metrics.mapping_precision,
      mapping_recall: score.metrics.mapping_recall,
      mapping_f1: score.metrics.mapping_f1,
      abstention_recall: score.metrics.abstention_recall,
      false_positives: score.counts.false_positive,
      false_positive_rate: score.metrics.false_positive_rate,
      wrong_maps: score.counts.wrong_map,
      missed_maps: score.counts.missed_map,
      cost_usd,
      latency_ms,
    }))
    .sort((a, b) => a.false_positives - b.false_positives || (b.mapping_f1 ?? 0) - (a.mapping_f1 ?? 0));
}

export function formatReport(name, score) {
  const m = score.metrics;
  const pct = (x) => (x === null ? "n/a" : `${(x * 100).toFixed(1)}%`);
  const lines = [
    `Predictor: ${name}   cases: ${score.total}`,
    `  accuracy ${pct(m.accuracy)} | mapping P ${pct(m.mapping_precision)} R ${pct(m.mapping_recall)} F1 ${pct(m.mapping_f1)}`,
    `  abstention P ${pct(m.abstention_precision)} R ${pct(m.abstention_recall)} | false-positive rate ${pct(m.false_positive_rate)} (${score.counts.false_positive} cases)`,
    `  outcomes: ${OUTCOMES.map((o) => `${o}=${score.counts[o]}`).join(" ")}`,
  ];
  for (const [cat, v] of Object.entries(score.false_positives_by_category)) {
    lines.push(`  false positives [${cat}]: ${v.false_positives}/${v.must_abstain_cases}`);
  }
  for (const f of score.failures) {
    lines.push(`  x ${f.id} (${f.category}): expected ${f.expected ?? "ABSTAIN"}, got ${f.predicted ?? "ABSTAIN"} -> ${f.outcome}`);
  }
  if (score.missing_predictions.length > 0) lines.push(`  missing predictions: ${score.missing_predictions.join(", ")}`);
  return lines.join("\n");
}

// ---------------------------------------------------------------
// Inputs from this repository
// ---------------------------------------------------------------

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..", "..");

export function loadBenchmark(file = path.join(HERE, "benchmark.json")) {
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  return data;
}

// The skill taxonomy (name +
// one-sentence definition), from data/capabilities.json.
// Ids are the capability names so predictions are human-readable.
export function loadCanonicalCatalog(file = path.join(REPO_ROOT, "data", "capabilities.json")) {
  return JSON.parse(fs.readFileSync(file, "utf8")).map((c) => ({ id: c.name, ...c }));
}

// Structural checks a benchmark must pass before its scores mean
// anything: unique ids, every expected label is a real canonical
// capability, every declared category is used, every must-abstain
// category is represented.
export function validateBenchmark(benchmark, catalog) {
  const problems = [];
  const names = new Set(catalog.map((c) => c.name));
  const ids = new Set();
  for (const c of benchmark.cases) {
    if (ids.has(c.id)) problems.push(`duplicate case id ${c.id}`);
    ids.add(c.id);
    if (!c.requirement || !String(c.requirement).trim()) problems.push(`${c.id}: empty requirement`);
    if (c.expected != null && !names.has(c.expected)) problems.push(`${c.id}: expected "${c.expected}" is not a canonical capability`);
    if (!(c.category in benchmark.categories)) problems.push(`${c.id}: unknown category ${c.category}`);
  }
  for (const cat of Object.keys(benchmark.categories)) {
    if (!benchmark.cases.some((c) => c.category === cat)) problems.push(`category ${cat} has no cases`);
  }
  return problems;
}
