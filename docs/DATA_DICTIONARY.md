# Data dictionary

The JSON/JSONL exports use the same names as the Python dataclasses. Enum
values are serialized as lower-case strings. Timestamps are informational; the
dataset hash excludes generated timestamps.

## Dataset and manifest

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Human-facing dataset name. |
| `cases` | array | Benchmark cases. |
| `definitions` | array | Benchmark definitions referenced by `definition_id`. |
| `version` | object | `benchmark_version`, `seed`, `generator_version`, `config`, `dataset_hash`, and `case_count`. |
| `metadata` | object | Provenance and synthetic/provider markers. |
| `statistics` | object/null | Derived counts and distributions. |

## Benchmark definition

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Definition name. |
| `capability` | string | Capability under test (for example `long-term-memory`). |
| `description` | string | Human-readable purpose. |
| `difficulty` | integer 1–5 | Default difficulty level. |
| `task_type` | string | `extraction`, `classification`, `memory`, `planning`, `tool_selection`, `arithmetic`, `structured_output`, or `instruction_following`. |
| `input_schema` | object | Shape of input/context supplied to a provider. |
| `expected_output` | any/object | Output shape or documentation. |
| `scoring` | object | Metric and task-specific scoring configuration. |
| `constraints` | object | Generation/evaluation constraints. |
| `definition_id` | string | Stable reference used by cases. |

## Benchmark case

| Field | Type | Meaning |
|---|---|---|
| `case_id` | string | Stable unique identifier. |
| `definition_id` | string | Definition this case instantiates. |
| `split` | enum | `train`, `validation`, `test`, or `challenge`. |
| `difficulty` | integer 1–5 | Case difficulty. Must correspond to concrete parameters. |
| `difficulty_params` | object | Context length, distractor count, fact count, conflicts, time distance, and task-specific knobs. |
| `task_type` | string | Benchmark type. |
| `prompt` | string | Instruction presented to a model. |
| `context` | any | Context presented to a model. |
| `input_data` | any/null | Structured input when applicable. |
| `expected_output` | any | Public/documentation copy of expected output. |
| `ground_truth` | any | Deterministic answer used for evaluation. Required for formal cases. |
| `scoring` | object | Case-level metric configuration. |
| `distractors` | array | Irrelevant, contradictory, near-match, long-context, missing-data, or tool-failure material. |
| `generator` | object | Template, rules, procedural generator, derived seed, version, and config. |
| `validation_status` | enum | `pending`, `valid`, `invalid`, or `warning`. |
| `validation_errors` | array | Human-readable validation failures/warnings. |
| `synthetic` | boolean | True for generated benchmark data. |
| `model_outputs` | array | Provider outputs; never ground truth. |
| `evaluation_results` | array | Scores/metrics for provider outputs. |
| `metadata` | object | Generator provenance, answer hash, and task-specific evidence. |

## Provider output and evaluation

`model_outputs[]` records `provider`, `model`, `output`, `latency_ms`, `tokens`,
`error`, and metadata. `provider` must identify `MockModel`, `RuleModel`, or a
future real provider. `evaluation_results[]` records `case_id`, `provider`,
metric values, aggregate score, pass/fail, latency, token counts, and details.

## Provenance rules

1. Synthetic cases carry `synthetic: true` and `metadata.data_origin` (or an
   equivalent generator marker).
2. Model outputs carry the provider name and provider type.
3. Ground truth is generated from rules/templates and is never copied from a
   model output.
4. A case with ambiguous, impossible, leaked, broken, or missing truth is not a
   valid formal benchmark case.

