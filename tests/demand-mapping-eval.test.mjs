// Tests for the evaluation harness (eval/demand-mapping).
//
// Two kinds of assertions:
//   1. The scorer computes exactly the right numbers on hand-checked
//      inputs (outcome classification, precision/recall/F1, abstention
//      metrics, false positives reported separately, regression gate).
//   2. The committed benchmark and baseline are trustworthy: every label
//      is a real canonical capability, every must-abstain category is
//      represented, the real American AI posting cases are present, and
//      re-running the deterministic gated predictor today reproduces the
//      committed baseline exactly (so the baseline was produced by a run,
//      not typed in).
// No network: live model predictors are never invoked here.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {
  classifyOutcome,
  compareToBaseline,
  comparePredictors,
  loadBenchmark,
  loadCanonicalCatalog,
  scorePredictions,
  validateBenchmark,
} from "../eval/demand-mapping/scorer.mjs";
import { BASELINE_FILE, GATED_PREDICTOR, isNameOnly, keywordOverlap, runLivePredictor, runOfflinePredictors, withNameGuard } from "../eval/demand-mapping/run.mjs";
import { buildSchema, costUsd, extractOutputText } from "../eval/demand-mapping/live.mjs";

let passed = 0, failed = 0;
async function test(name, fn) {
  try { await fn(); console.log(`PASS  ${name}`); passed++; }
  catch (err) { console.log(`FAIL  ${name}: ${err.message}`); failed++; }
}

const CASES = [
  { id: "m1", expected: "Evaluation", category: "genuine_fit" },
  { id: "m2", expected: "RAG", category: "genuine_fit" },
  { id: "m3", expected: "CI/CD", category: "genuine_fit" },
  { id: "a1", expected: null, category: "technology_name_only" },
  { id: "a2", expected: null, category: "technology_name_only" },
  { id: "a3", expected: null, category: "ontology_gap" },
];

await test("every case lands in exactly one outcome: correct_map, wrong_map, missed_map, correct_abstain, false_positive", () => {
  assert.equal(classifyOutcome("Evaluation", "Evaluation"), "correct_map");
  assert.equal(classifyOutcome("Evaluation", "RAG"), "wrong_map");
  assert.equal(classifyOutcome("Evaluation", null), "missed_map");
  assert.equal(classifyOutcome(null, null), "correct_abstain");
  assert.equal(classifyOutcome(null, "CI/CD"), "false_positive");
  assert.equal(classifyOutcome(undefined, undefined), "correct_abstain");
});

await test("metrics are exact on a hand-checked prediction set", () => {
  // m1 correct, m2 wrong (-> Tool Use), m3 missed; a1 correct abstain,
  // a2 false positive (the "Strong Python -> CI/CD" failure), a3 correct.
  const s = scorePredictions(CASES, { m1: "Evaluation", m2: "Tool Use", m3: null, a1: null, a2: "CI/CD", a3: null });
  assert.deepEqual(s.counts, { correct_map: 1, wrong_map: 1, missed_map: 1, correct_abstain: 2, false_positive: 1 });
  assert.equal(s.metrics.accuracy, 0.5);
  assert.equal(s.metrics.mapping_precision, 0.3333, "1 correct of 3 predicted mappings");
  assert.equal(s.metrics.mapping_recall, 0.3333, "1 correct of 3 expected mappings");
  assert.equal(s.metrics.mapping_f1, 0.3333);
  assert.equal(s.metrics.abstention_precision, 0.6667, "2 correct of 3 abstentions");
  assert.equal(s.metrics.abstention_recall, 0.6667, "2 of 3 must-abstain cases abstained");
  assert.equal(s.metrics.false_positive_rate, 0.3333);
  assert.deepEqual(s.false_positives_by_category, {
    technology_name_only: { false_positives: 1, must_abstain_cases: 2, rate: 0.5 },
    ontology_gap: { false_positives: 0, must_abstain_cases: 1, rate: 0 },
  });
  assert.deepEqual(s.failures.map((f) => f.id), ["m2", "m3", "a2"]);
});

await test("false positives are reported separately from ordinary mapping errors", () => {
  const s = scorePredictions(CASES, { m1: "Evaluation", m2: "RAG", m3: "CI/CD", a1: "Evaluation", a2: null, a3: null });
  assert.equal(s.counts.false_positive, 1);
  assert.equal(s.counts.wrong_map, 0);
  assert.equal(s.metrics.mapping_recall, 1);
  assert.ok(s.metrics.mapping_precision < 1, "a false positive still costs mapping precision");
});

await test("edge cases: no predicted mappings -> precision is n/a (null), never a fake 0 or 1; a missing prediction is scored as abstention and listed", () => {
  const s = scorePredictions(CASES, { m1: null, m2: null, m3: null, a1: null, a2: null });
  assert.equal(s.metrics.mapping_precision, null);
  assert.equal(s.metrics.mapping_recall, 0);
  assert.equal(s.metrics.mapping_f1, 0);
  assert.deepEqual(s.missing_predictions, ["a3"]);
});

await test("regression gate: identical run passes; a metric drop fails; a single traded-off case fails even when totals are flat", () => {
  const base = scorePredictions(CASES, { m1: "Evaluation", m2: "RAG", m3: null, a1: null, a2: null, a3: null });
  assert.equal(compareToBaseline(base, base).passed, true);

  const worse = scorePredictions(CASES, { m1: "Evaluation", m2: null, m3: null, a1: null, a2: null, a3: null });
  const g1 = compareToBaseline(worse, base);
  assert.equal(g1.passed, false);
  assert.ok(g1.regressions.some((r) => r.kind === "metric" && r.metric === "mapping_recall"));
  assert.ok(g1.regressions.some((r) => r.kind === "case" && r.id === "m2"));

  // Same totals, different cases: m2 lost, m3 gained.
  const traded = scorePredictions(CASES, { m1: "Evaluation", m2: null, m3: "CI/CD", a1: null, a2: null, a3: null });
  const g2 = compareToBaseline(traded, base);
  assert.equal(g2.passed, false, "per-case regression is caught even though mapping totals are unchanged");
  assert.ok(g2.regressions.some((r) => r.kind === "case" && r.id === "m2"));
  assert.ok(g2.improvements.some((r) => r.kind === "case" && r.id === "m3"));
});

await test("regression gate: a NEW false positive is always a regression, even within tolerance", () => {
  const base = scorePredictions(CASES, { m1: "Evaluation", m2: "RAG", m3: "CI/CD", a1: null, a2: null, a3: null });
  const fp = scorePredictions(CASES, { m1: "Evaluation", m2: "RAG", m3: "CI/CD", a1: null, a2: "CI/CD", a3: null });
  const gate = compareToBaseline(fp, base, { tolerance: 0.5 });
  assert.equal(gate.passed, false);
  assert.ok(gate.regressions.some((r) => r.kind === "case" && r.id === "a2" && r.current === "false_positive"));
});

await test("predictor comparison ranks fewest false positives first, then mapping F1", () => {
  const safe = scorePredictions(CASES, { m1: "Evaluation", m2: null, m3: null, a1: null, a2: null, a3: null });
  const greedy = scorePredictions(CASES, { m1: "Evaluation", m2: "RAG", m3: "CI/CD", a1: "Evaluation", a2: "CI/CD", a3: "RAG" });
  const rows = comparePredictors([{ name: "greedy", score: greedy }, { name: "safe", score: safe }]);
  assert.deepEqual(rows.map((r) => r.predictor), ["safe", "greedy"]);
  assert.equal(rows[1].false_positives, 3);
});

// ---------------------------------------------------------------
// The committed benchmark and baseline
// ---------------------------------------------------------------
const benchmark = loadBenchmark();
const catalog = loadCanonicalCatalog();

await test("the skill taxonomy has 40 skills, each with a definition", () => {
  assert.equal(catalog.length, 40);
  assert.ok(catalog.every((c) => c.name && c.description.length > 20));
  assert.ok(catalog.some((c) => c.name === "Evaluation" && /evals, test sets, benchmarks, graders/.test(c.description)));
});

await test("benchmark is structurally valid: unique ids, every label is a canonical capability, every category is used", () => {
  assert.deepEqual(validateBenchmark(benchmark, catalog), []);
  const broken = { ...benchmark, cases: [...benchmark.cases, { id: "x", requirement: "Strong Python", expected: "Python", category: "genuine_fit" }] };
  assert.ok(validateBenchmark(broken, catalog).some((p) => /not a canonical capability/.test(p)), "an ontology-gap label can never sneak in as a mapping target");
});

await test("benchmark covers both sides: genuine fits AND every must-abstain category, including the real American AI posting requirements", () => {
  const abstain = benchmark.cases.filter((c) => c.expected === null);
  const mapped = benchmark.cases.filter((c) => c.expected !== null);
  assert.ok(mapped.length >= 20 && abstain.length >= 12);
  for (const cat of ["technology_name_only", "proficiency_phrasing", "ontology_gap", "near_miss"]) {
    assert.ok(abstain.some((c) => c.category === cat), cat);
  }
  const real = Object.fromEntries(benchmark.cases.filter((c) => c.source === "american_ai_public_posting").map((c) => [c.requirement, c.expected]));
  assert.deepEqual(real, { evals: "Evaluation", "Strong Python": null, LLMs: null, "track record of taking models from notebook to live traffic": null });
  assert.ok(benchmark.cases.some((c) => /^para-/.test(c.id)), "paraphrase cases guard against a benchmark that only rewards vocabulary overlap");
});

await test("name-only rule: bare technology names abstain, requirements that describe work do not", () => {
  for (const r of ["Strong Python", "LLMs", "TypeScript", "Docker and Kubernetes", "5+ years of PostgreSQL", "hands-on experience with PyTorch"]) {
    assert.equal(isNameOnly(r), true, r);
  }
  for (const r of ["maintain GitHub Actions build, test, and deploy pipelines", "design and version REST APIs consumed by partner integrations", "deploy services on AWS"]) {
    assert.equal(isNameOnly(r), false, r);
  }
  const cases = [{ id: "t", requirement: "Strong Python" }, { id: "f", requirement: "maintain GitHub Actions build, test, and deploy pipelines" }];
  assert.deepEqual(withNameGuard(cases, { t: "CI/CD", f: "CI/CD" }), { t: null, f: "CI/CD" }, "the guard only ever turns a mapping into an abstention");
});

await test("the committed baseline was produced by an actual run: re-running the deterministic gated predictor reproduces it exactly and the gate passes", () => {
  const baseline = JSON.parse(fs.readFileSync(BASELINE_FILE, "utf8"));
  assert.equal(baseline.predictor, GATED_PREDICTOR);
  const gated = runOfflinePredictors(benchmark, catalog).find((r) => r.name === GATED_PREDICTOR);
  assert.deepEqual(gated.score.metrics, baseline.metrics);
  assert.deepEqual(gated.score.per_case, baseline.per_case);
  assert.equal(compareToBaseline(gated.score, baseline).passed, true);
});

await test("the harness exposes the failure modes it exists to catch: a rule-based guard removes some false positives but not all, and costs a genuine fit", () => {
  const results = Object.fromEntries(runOfflinePredictors(benchmark, catalog).map((r) => [r.name, r.score]));
  assert.ok(results["keyword-overlap"].counts.false_positive > 0);
  assert.ok(results[GATED_PREDICTOR].counts.false_positive < results["keyword-overlap"].counts.false_positive, "the guard removes some false positives");
  assert.ok(results[GATED_PREDICTOR].counts.false_positive > 0, "but semantic abstention cannot be done by a keyword rule alone");
  assert.equal(results[GATED_PREDICTOR].per_case.find((p) => p.id === "aa-evals").outcome, "missed_map", "the rule cannot tell a practice name ('evals') from a tool name");
  assert.equal(results["abstain-all"].metrics.false_positive_rate, 0);
  assert.equal(results["abstain-all"].metrics.mapping_recall, 0);
  assert.ok(keywordOverlap([{ id: "p", requirement: "make sure a customer is never billed twice when a request is resent" }], catalog).p !== "Idempotency", "paraphrases defeat vocabulary matching");
});

await test("live predictor parses the API response, keeps the output inside the taxonomy, and totals cost -- with a stubbed API, no network", async () => {
  const replies = { "aa-evals": "Evaluation", "aa-strong-python": "Not A Real Skill" };
  const fakeFetch = async (_url, init) => {
    const req = JSON.parse(init.body);
    const id = benchmark.cases.find((c) => req.input.includes(`"""${c.requirement}"""`)).id;
    const skill = replies[id] ?? null;
    return {
      ok: true,
      json: async () => ({
        output: [{ type: "message", content: [{ type: "output_text", text: JSON.stringify({ skill, reason: "stub" }) }] }],
        usage: { input_tokens: 1000, output_tokens: 20 },
      }),
    };
  };
  const live = await runLivePredictor(benchmark, catalog, "gpt-4o-mini", "test-key", fakeFetch);
  assert.equal(live.predictions["aa-evals"], "Evaluation");
  assert.equal(live.predictions["aa-strong-python"], null, "a label outside the taxonomy is scored as abstention");
  assert.equal(live.input_tokens, 1000 * benchmark.cases.length);
  assert.equal(live.cost_usd, costUsd("gpt-4o-mini", 1000 * benchmark.cases.length, 20 * benchmark.cases.length));
  assert.equal(extractOutputText({ output: [{ content: [{ type: "output_text", text: "x" }] }] }), "x");
  assert.deepEqual(buildSchema(catalog).properties.skill.enum.at(-1), null, "abstention is an allowed answer in the schema");
  await assert.rejects(runLivePredictor(benchmark, catalog, "gpt-4o-mini", undefined, fakeFetch), /OPENAI_API_KEY/);
});

await test("live model runs are opt-in only: they require OPENAI_API_KEY and are never invoked by these tests", () => {
  const src = fs.readFileSync(path.join(process.cwd(), "eval", "demand-mapping", "run.mjs"), "utf8");
  assert.match(src, /if \(!apiKey\) throw new Error\("OPENAI_API_KEY is required for --live"\)/);
  assert.match(src, /if \(argv\.includes\("--live"\)\)/);
  assert.ok(!fs.existsSync(path.join(process.cwd(), "eval", "demand-mapping", "results")) || fs.readdirSync(path.join(process.cwd(), "eval", "demand-mapping", "results")).every((f) => f.startsWith("live-")));
});

console.log(`\n${passed}/${passed + failed} tests passed.`);
if (failed > 0) process.exit(1);
