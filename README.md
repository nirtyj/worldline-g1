# Worldline on G1

Worldline's runtime driving a Unitree G1 humanoid in an Isaac Sim 5.1 household (AllenAI MolmoSpaces
ProcTHOR houses), walking with NVIDIA GEAR-SONIC. The robot/tool API follows a public reconstruction of
Ludi 0.1's robot API. The full plan is [`PLAN.md`](PLAN.md); read §0 (owner decisions) first.

**Where it stands (M2a).** M1 is done: SONIC stands, walks, turns and stops the G1 in `procthor-train-38`,
driven through our own `BodyClient` ([`docs/M1.md`](docs/M1.md)). M2a moves Worldline's runtime onto the new
API and runs it on the `lite` profile (pure Python, recorded house), offline. The Isaac profiles
(`bringup`, `sonic`, `full`) come in M2b.

## The pieces

| Dir | What | Runs in |
|---|---|---|
| `api/` | **The contract**, stdlib only: the tools (`tools.py`, the only definition), result envelope and typed results, reason codes, summaries, execution objects with ids, generations and epochs, context events, the tool-state machine, observations, service Protocols, the skill registry, common types. `python -m api.gen_schemas` writes `api/schemas/*.json` | py3.11 runtime, py3.12 body server, py3.11 Isaac |
| `agent/` | The harness: validation pipeline (`validate.py`: SCHEMA, ENUM, STATE, CAPABILITY), belief vs truth, layout and goal check, memory, recall, procedures, persona, narrator, mutants, planner prompt (`model.py`), context bus | P5 `wl-runtime` |
| `brains/` | System 1 (Jev labels + Gemini Live observer, frame gate with G1 presets), the composite brain, the brain interface | P5 |
| `llmkit/` | Tool-calling clients (Anthropic, Gemini, OpenAI-compatible); schemas come from `api/tools.py` | P5 |
| `sim/` | The one clock and the ground-truth event log (the runtime never reads the log) | P5 |
| `robot/`, `services/`, `world/`, `config/` | `RobotBridge` (`G1Robot`), the services behind the tools, the world model (the only ground-truth reader), profiles | P5 |
| `ui/`, `eval/` | The live web UI and the 17-scenario eval | P5 |
| `sim_isaac/`, `body/`, `scenes/`, `sonic/`, `nav2/`, `viz/` | M1: Isaac app + DDS bridge + GT server, the SONIC body server, MolmoSpaces loader, deploy tooling, visual recorder | P1, P3, box |

## Tools the planner sees

`speak`, `list_locations`, `navigate`, `check_reachability`, `manipulate(pick|place)`,
`wait_and_observe`, and `recall` [WL]. Looking is automatic: a scan when the robot arrives at a keypoint,
a glance after every manipulate, and a look at the start of every `wait_and_observe`. Every result is a
frozen `ToolResult` envelope (`succeeded / failed / cancelled / timed_out / rejected`) that carries the
latest `observation_id`; a rejection is a tool result too. Stepping-stone executors (kinematic, arm script
with attach grasp, lite) are labelled `[fallback]` in results, prompts and eval.

## Run the tests (laptop, offline)

```bash
uv venv --python 3.11 .venv-rt
uv pip install --python .venv-rt/bin/python numpy scipy pillow pyzmq msgpack pytest \
    "google-genai==2.25.0" "typesafe_sdk==0.7.2" pyyaml
.venv-rt/bin/python -m pytest                       # tests/unit, tests/contract, tests/kept
.venv-rt/bin/python -m pytest tests/world tests/services tests/ui tests/eval   # the other agents' suites
.venv-rt/bin/python -m api.gen_schemas --check      # api/schemas/*.json is current
```

No test calls a model. The opt-in live checks need keys (`.env.example` lists them):

```bash
.venv-rt/bin/python -m pytest -m live tests/contract/test_live_planner.py
.venv-rt/bin/python -m tests.kept.system1_live_check
```

## Keys

Copy `.env.example` to `.env` (gitignored) or `~/.config/ludo-g1/secrets.env`. Lookup order:
`$WORLDLINE_ENV`, `<repo>/.env`, `~/.config/ludo-g1/secrets.env`. `ludo-runtime/.env` is never read.

## Provenance

Imported from `ludo-runtime@cb4ce53` (read-only); see [`docs/PROVENANCE`](docs/PROVENANCE) for what was kept,
adapted, rewritten or left out.
