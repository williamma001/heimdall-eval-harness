#!/usr/bin/env node
// Runs predictors over the benchmark, compares them, and gates against a
// committed baseline.
//
//   node eval/demand-mapping/run.mjs                 offline predictors + regression check
//   node eval/demand-mapping/run.mjs --write-baseline
//   OPENAI_API_KEY=... node eval/demand-mapping/run.mjs --live --model gpt-4o-mini --model gpt-4o
//
// Offline predictors run anywhere (CI included) with no network:
//   abstain-all                  floor: perfect abstention, zero mapping recall
//   keyword-overlap              naive "topically close" mapper -- the failure
//                                mode the benchmark exists to expose
//   keyword-overlap+name-guard   the same decisions passed through a simple
//                                rule that abstains on bare technology names,
//                                to measure what a rule-based filter can and
//                                cannot fix
//
// Live predictors (--live) call an OpenAI model once per case. They run
// only when the caller supplies OPENAI_API_KEY; their results are written
// to results/ only from an actual run.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { LIVE_MODELS, classifyRequirement, costUsd } from "./live.mjs";
import { compareToBaseline, comparePredictors, formatReport, loadBenchmark, loadCanonicalCatalog, scorePredictions, validateBenchmark } from "./scorer.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const BASELINE_FILE = path.join(HERE, "baseline.json");
export const GATED_PREDICTOR = "keyword-overlap+name-guard";

// ---------------------------------------------------------------
// Offline predictors
// ---------------------------------------------------------------

const STOPWORDS = new Set([
  "a", "an", "and", "or", "the", "of", "to", "in", "on", "for", "with", "by", "as", "is", "are", "be", "that", "this", "it", "its",
  "from", "at", "into", "their", "your", "our", "not", "no", "per", "use", "using", "like", "such", "can", "so", "we", "you",
  "experience", "years", "strong", "build", "building", "work", "working", "ability", "able",
]);

export function contentTokens(text) {
  return String(text ?? "")
    .toLowerCase()
    .split(/[^a-z0-9+#]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t))
    .map((t) => (t.length > 4 && t.endsWith("s") ? t.slice(0, -1) : t));
}

export function abstainAll(cases) {
  return Object.fromEntries(cases.map((c) => [c.id, null]));
}

// Picks the skill whose name + definition shares the most content tokens
// with the requirement; abstains only on zero overlap. Ties go to the
// catalog's first entry. Deliberately naive.
export function keywordOverlap(cases, catalog) {
  const capTokens = catalog.map((cap) => ({ name: cap.name, tokens: new Set(contentTokens(`${cap.name} ${cap.description}`)) }));
  return Object.fromEntries(
    cases.map((c) => {
      const req = contentTokens(c.requirement);
      let bestName = null;
      let bestScore = 0;
      for (const cap of capTokens) {
        const score = req.filter((t) => cap.tokens.has(t)).length;
        if (score > bestScore) {
          bestScore = score;
          bestName = cap.name;
        }
      }
      return [c.id, bestName];
    }),
  );
}

// Words that qualify a technology name without describing any work.
const QUALIFIERS = new Set([
  "strong", "solid", "deep", "expert", "expertise", "proficient", "proficiency", "hands", "on", "experience", "experienced",
  "knowledge", "familiarity", "familiar", "with", "in", "of", "and", "or", "years", "year", "plus",
]);
// Verbs that signal a requirement describes work rather than naming a tool.
const WORK_VERBS = new Set([
  "design", "build", "implement", "run", "deploy", "ship", "lead", "make", "model", "instrument", "maintain", "fine", "tune",
  "optimize", "write", "own", "debug", "secure", "evaluate", "route", "orchestrate", "integrate", "monitor", "manage", "develop",
]);

// True when a requirement is only a technology name, possibly with
// proficiency wording around it ("Strong Python", "5+ years of PostgreSQL").
export function isNameOnly(requirement) {
  const words = String(requirement ?? "").toLowerCase().split(/[^a-z0-9+#.]+/).filter(Boolean);
  if (words.some((w) => WORK_VERBS.has(w))) return false;
  const rest = words.filter((w) => !QUALIFIERS.has(w) && !/^\d+\+?$/.test(w));
  return rest.length > 0 && rest.length <= 3;
}

// Applies the name-only rule on top of any predictor's raw decisions.
export function withNameGuard(cases, rawPredictions) {
  return Object.fromEntries(cases.map((c) => [c.id, isNameOnly(c.requirement) ? null : (rawPredictions[c.id] ?? null)]));
}

export function runOfflinePredictors(benchmark, catalog) {
  const cases = benchmark.cases;
  const raw = keywordOverlap(cases, catalog);
  return [
    { name: "abstain-all", predictions: abstainAll(cases) },
    { name: "keyword-overlap", predictions: raw },
    { name: GATED_PREDICTOR, predictions: withNameGuard(cases, raw) },
  ].map((p) => ({ ...p, score: scorePredictions(cases, p.predictions) }));
}

// ---------------------------------------------------------------
// Live predictor (optional; needs OPENAI_API_KEY)
// ---------------------------------------------------------------

export async function runLivePredictor(benchmark, catalog, model, apiKey, fetchImpl) {
  if (!apiKey) throw new Error("OPENAI_API_KEY is required for --live");
  if (!LIVE_MODELS.includes(model)) throw new Error(`--model must be one of ${LIVE_MODELS.join(", ")}`);
  const predictions = {};
  const reasons = {};
  let inputTokens = 0;
  let outputTokens = 0;
  const started = Date.now();
  for (const c of benchmark.cases) {
    const r = await classifyRequirement({ requirement: c.requirement, catalog, model, apiKey, fetchImpl });
    predictions[c.id] = r.skill;
    reasons[c.id] = r.reason;
    inputTokens += r.inputTokens;
    outputTokens += r.outputTokens;
  }
  return {
    name: `live:${model}`,
    predictions,
    reasons,
    score: scorePredictions(benchmark.cases, predictions),
    cost_usd: costUsd(model, inputTokens, outputTokens),
    latency_ms: Date.now() - started,
    input_tokens: inputTokens,
    output_tokens: outputTokens,
  };
}

// ---------------------------------------------------------------
// CLI
// ---------------------------------------------------------------

function snapshot(entry) {
  return { predictor: entry.name, metrics: entry.score.metrics, counts: entry.score.counts, per_case: entry.score.per_case };
}

async function main(argv) {
  const benchmark = loadBenchmark();
  const catalog = loadCanonicalCatalog();
  const problems = validateBenchmark(benchmark, catalog);
  if (problems.length > 0) {
    console.error(`Benchmark is invalid:\n  ${problems.join("\n  ")}`);
    return 2;
  }

  const results = runOfflinePredictors(benchmark, catalog);
  if (argv.includes("--live")) {
    const models = argv.flatMap((a, i) => (a === "--model" ? [argv[i + 1]] : []));
    for (const model of models.length > 0 ? models : ["gpt-4o-mini"]) {
      const live = await runLivePredictor(benchmark, catalog, model, process.env.OPENAI_API_KEY);
      results.push(live);
      const outDir = path.join(HERE, "results");
      fs.mkdirSync(outDir, { recursive: true });
      const file = path.join(outDir, `live-${model}-${new Date().toISOString().slice(0, 10)}.json`);
      const record = { ...snapshot(live), cost_usd: live.cost_usd, latency_ms: live.latency_ms, input_tokens: live.input_tokens, output_tokens: live.output_tokens, reasons: live.reasons };
      fs.writeFileSync(file, `${JSON.stringify(record, null, 2)}\n`);
      console.log(`wrote ${path.relative(process.cwd(), file)}`);
    }
  }

  for (const r of results) console.log(`\n${formatReport(r.name, r.score)}`);
  console.log("\nComparison (fewest false positives first):");
  console.table(comparePredictors(results));

  const gated = results.find((r) => r.name === GATED_PREDICTOR);
  if (argv.includes("--write-baseline")) {
    fs.writeFileSync(BASELINE_FILE, `${JSON.stringify(snapshot(gated), null, 2)}\n`);
    console.log(`\nwrote baseline for ${GATED_PREDICTOR}`);
    return 0;
  }
  if (!fs.existsSync(BASELINE_FILE)) {
    console.error("\nno baseline.json -- run with --write-baseline first");
    return 1;
  }
  const baseline = JSON.parse(fs.readFileSync(BASELINE_FILE, "utf8"));
  const gate = compareToBaseline(gated.score, baseline);
  console.log(`\nRegression gate (${GATED_PREDICTOR} vs baseline): ${gate.passed ? "PASS" : "FAIL"}`);
  for (const r of gate.regressions) console.log(`  regression: ${JSON.stringify(r)}`);
  for (const i of gate.improvements) console.log(`  improvement: ${JSON.stringify(i)}`);
  return gate.passed ? 0 : 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (err) => {
      console.error(err instanceof Error ? err.message : err);
      process.exit(1);
    },
  );
}
