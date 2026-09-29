# Robot Harness Landscape

> **Historical (THOR era).** Written for Worldline on AI2-THOR (ludo-runtime@cb4ce53), before the G1 port. Kept as background; where it disagrees with PLAN.md or docs/M2.md, those win.

*Research, September 2026, for this robot runtime. Page version: https://claude.ai/artifact/2SsznG6FoPXGVkpjF2c79j*

*Written when the runtime still had voice (Gemini 3.8 Live as the ears, browser speech) and Claude as the planner. Since then voice was removed (chat is typed, the planner classifies it, replies appear as text), the planner is Gemini 3.8 Flash, and several roadmap items were built: the roadmap below says where each stands. The System 1 / System 2 discussion still applies to a robot that has voice.*

What the field does today with systems like this runtime: a model that hears, a model
that plans, a runtime that checks, and a body that acts. It covers robot
foundation models, agent harnesses, memory, safety and human–robot interaction,
and ends with 13 changes for the runtime, ordered by what to do first.

## The short version

The runtime's shape is where the field has landed: a fast model for the human side, a
slower planner, and deterministic code that owns execution and safety. The
biggest gaps are in how it talks and listens, how it finds things, how much it
sends the planner, and the lack of a way to measure any of it. The first five
changes are small:

1. **Stop talking when you start.** Pause or adapt the robot's speech the moment
   you speak, not only on "stop".
2. **Say what it can and can't do, and why it's doing something.** Say it aloud,
   rather than relying on you to read the panel.
3. **Search by expected cost.** The planner rates how likely each spot is, and code
   picks the order using travel cost.
4. **Put the prompt on a diet.** Send what matters now, and give the planner a
   `recall` tool for the rest.
5. **Ask only when two things really fit.** Measure the ambiguity instead of
   leaving it to the prompt.

## Where the runtime already matches the field

| | |
|---|---|
| **Governance outside the model** | Rules are checked by code at the moment each decision arrives. Recent governance work argues for exactly this separation. |
| **A stop that skips every model** | It halts the motors first, then sends the acknowledgement, then settles the cancel, then lets the planner talk. It is the kind of human override the governance work treats as its own layer. |
| **Two speeds of thinking** | Fast ears and a slow planner, like the System 1 / System 2 split in Helix, GR00T N1 and Gemini Robotics. |
| **Belief with provenance, verified after acting** | Every value records its source, time and whether it was verified, and every pick and place is checked by a look. This is the motivation behind the success-detection work. |
| **Stale decisions are dropped** | Request versions and control epochs catch the problem the asynchronous-planning papers point to: a plan made for a world that has since changed. |
| **A persona that defaults to medium** | In a CHI 2019 study, medium proactivity was rated more helpful than high, and people preferred it. |

## Where the 13 changes land

Each `(n)` is one change from the roadmap below, placed where it would live in
the runtime. Most touch the runtime itself, which is also where the field puts the work.

```
                        ┌──────────────── YOU · the page ────────────────┐
                        │ speak, type, watch                             │
                        └──────────┬───────────────────────▲─────────────┘
                          what you say                what it says
┌─ GEMINI · the ears ─┐ ┌──────────▼── RUNTIME · agent/ ────┴─────────────┐ ┌─ PERSONA ───────────────────┐
│ labels each turn    │ │ (1)  speech: yield or adapt when you talk       │ │ (11) learns what you accept │
│ partial "stop" →    ├─► (3)  search: likely spots × travel cost        ◄─┤ levels · drives · own goals │
│ halt                │ │ (4)  prompt: what matters + a recall tool       │ └─────────────────────────────┘
└─────────────────────┘ │ (8)  rules as contracts, live and on traces     │ ┌─ MEMORY ────────────────────┐
┌─ CLAUDE · planner ──┐ │ (10) think while moving: next step early        ◄─► (7)  what moves, what stays  │
│ (5)  asks when two  ◄─►                                                 │ │ (13) procedures learned     │
│      things fit     │ │ kept: stop lane · versions · belief · reconcile │ │      from corrections       │
│ (2)  says what it   │ └─────────────────────────▲───────────────────────┘ └─────────────────────────────┘
│      can't do       │                           │
│ (13) reads learned  │ ┌─ ROBOT · skills ────────▼───────────────────────┐
│      rules          │ │ (12) pick and place by a learned policy (VLA)   │
└─────────────────────┘ │ (9)  look: a visual check that it worked        │
                        └─────────────────────────────────────────────────┘
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
  (6) scenario suite with simulated users around all of it: success, time, calls, tokens,
      unasked actions, questions
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

Nothing here replaces a part of the runtime. The stop lane, versions, belief and
reconcile stay as they are; the changes add to them or sharpen them.

## What the field does, theme by theme

### 1. Two brains and a body

- [Helix](https://www.figure.ai/news/helix) (Figure) runs a 7B vision-language
  model at **7–9 Hz** for understanding, and an 80M visuomotor policy at
  **200 Hz** for control.
- [GR00T N1](https://arxiv.org/abs/2503.14734) (NVIDIA) uses the same split: the
  reasoning model at **10 Hz**, and a diffusion transformer generating motions.
- [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) pairs an
  embodied-reasoning model that plans and calls tools, including the action
  model, with an action model that reasons before it moves.
  [ER 1.6](https://deepmind.google/blog/gemini-robotics-er-1-6/) (April 2026) is
  offered to developers as the orchestrator for their own robot APIs.
- [Hi Robot](https://arxiv.org/abs/2502.19417) (Physical Intelligence) has a
  high-level model that turns open-ended prompts and mid-task feedback into
  steps the low-level policy can already do.

**For this runtime:** the split was the same, except its fast side was its ears
(Gemini), and its motor skills are scripted THOR calls. The runtime can stay
exactly where it is when a learned policy replaces those skills **(12)**.

### 2. The harness is half the result

- [Show-Harness](https://arxiv.org/abs/2609.10522) (September 2026) gives the
  model a compact set of semantic actions, turned into robot commands by
  deterministic code. Frontier models then control robots zero-shot and beat
  both action-model and agent baselines.
- [AgenticNav](https://arxiv.org/abs/2606.10577) (June 2026) exposes
  `move_to(pixel)`, `query_depth` and `recall` as tools, recalling past views on
  demand instead of piling up history. It reached **55%** success against
  **44%** for the same model behind a learned waypoint predictor, and **46.7%**
  against **23.3%** on a real robot.
- Phillip Isola's
  [robot-use agents](https://web.mit.edu/phillipi/www/writing/robot-use-agents.html)
  (September 2026) argues that LLMs will use robots through their APIs the way
  they already use computers.

**For this runtime:** it is one of these harnesses. The lesson is to let the planner ask
for state rather than dumping all of it into every prompt, which was then about
6k tokens a call **(4)**, and later to let it point at pixels **(9)**.

### 3. Governance lives outside the model

- [Runtime governance for embodied agents](https://arxiv.org/abs/2604.07833)
  (April 2026) makes policy checks, capability admission, monitoring, rollback
  and human override one separate layer. It intercepted **96.2%** of
  unauthorized actions, cut unsafe continuation from **100%** to **22%**, and
  recovered **90.7%** of the time.
- [ContrAgent](https://arxiv.org/abs/2609.18128) (September 2026) writes the
  rules as temporal-logic contracts over the trace of tool calls. Each compiles
  to an automaton that gates actions live and audits recorded runs afterwards,
  with deterministic verdicts, far faster than an LLM judge.
- [RoboGuard](https://arxiv.org/abs/2503.07885) grounds general safety rules in
  the current scene, then repairs unsafe plans. Under jailbreak attacks, unsafe
  plans executed fell from **92%** to under **3%**.
- [RoboPAIR](https://arxiv.org/abs/2410.13691) jailbroke LLM-driven robots with
  attack success often near **100%**, including a commercial robot dog.

**For this runtime:** the separation is already there. The next steps are to state the
rules as contracts that can be checked and replayed, and to add room-specific
safety rules. Anyone who can talk to the robot can try to talk it into
something **(8)**.

### 4. Memory: structure beats volume

- [SayPlan](https://arxiv.org/abs/2307.06135) lets the LLM search a collapsed 3D
  scene graph and expand only the parts it needs. It planned across 3 floors, 36
  rooms and 140 objects.
- [HOV-SG](https://arxiv.org/abs/2403.17846) organises memory as floor, room and
  object, **75%** smaller than a dense map.
  [ConceptGraphs](https://concept-graphs.github.io/) builds open-vocabulary
  object graphs from 2D models.
- [ReMEmbR](https://nvidia-ai-iot.github.io/remembr/) (NVIDIA) stores captions
  with position and time, and the LLM queries them step by step.
- [LT-Mem](https://arxiv.org/abs/2608.19059) (August 2026) decides per object
  whether to overwrite, hold, or keep several hypotheses, based on how often that
  kind of thing moves. It used an order of magnitude fewer tokens than the
  baselines.
- [Procedural Graphs](https://arxiv.org/abs/2609.09153) (September 2026) keeps
  "what to do next" knowledge as a graph that rewrites itself from failed runs.

**For this runtime:** belief with provenance is a good base. Missing are volatility (one
missed look shouldn't erase a kettle) and containers and rooms **(7)**,
query-on-demand **(4)**, and procedures learned from runs **(13)**.

### 5. Finding things

- [LLM-informed object search](https://arxiv.org/abs/2603.23800) (IROS 2026)
  treats the LLM as a fallible prior: it rates how likely each location is, and a
  planner weighs that against travel. This was up to **11.8%** cheaper than
  letting the LLM choose, and **39.2%** cheaper than an optimistic baseline.
- [Finder](https://arxiv.org/abs/2609.18058) (September 2026) runs a typed loop:
  plan where to look, gather evidence, verify the candidate, then accept,
  continue or give up. It gained **15.75** points in success within 1 m.

**For this runtime:** the planner picks where to search on its own, one call per spot. It
should rate the spots once, and let code choose the order **(3)**.

### 6. Knowing it worked

- [AHA](https://arxiv.org/abs/2410.00371) (NVIDIA) is a model trained to detect
  and explain manipulation failures. It beat GPT-4o by **10.3%**, and its
  feedback raised task success by **21.4%**.
- [Code-as-Monitor](https://arxiv.org/abs/2412.04455) has a vision-language
  model write constraint-checking code that watches for failures as they happen,
  or before.
- [Gemini Robotics-ER 1.5](https://arxiv.org/abs/2510.03342) reports multi-view
  success detection of **0.79–0.80**; ER 1.6 improves the multi-view reasoning.

**For this runtime:** today a look reads THOR's truth for what's in view. A visual check
from the head camera, graded against that truth, would show how far real
perception has to go **(9)**.

### 7. Latency and cost

- [AgenticCache](https://arxiv.org/abs/2604.24039) (MLSys 2026) caches frequent
  plan transitions and checks them against the state in the background. It cut
  latency **65%** and tokens **50%**, and raised success **22%**.
- [Thinking while driving](https://www.arxiv.org/pdf/2512.10610) starts the
  model's next decision while the current move runs, so the model's latency
  hides inside the motion.
- [Robotouille](https://arxiv.org/abs/2502.05227) shows how hard this is: a
  ReAct agent solved **47%** of synchronous tasks but only **11%** of
  asynchronous ones.
- A dialogue robot in a real mall answered with a short
  [context-aware preface](https://arxiv.org/abs/2607.23204) before the full
  reply, which cut the gap before its first words.

**For this runtime:** each step costs 1.5–2 s and about 6k tokens, and a searched spot
takes two calls. Asking for the next step while the robot drives, and caching
routine sequences like pick → look → go to the user, attack both. The
stale-decision check keeps it safe **(10)**.

### 8. The human in the loop

- **Asking.** [KnowNo](https://arxiv.org/abs/2307.01928) uses conformal
  prediction to get a set of plausible options with a guaranteed success rate,
  and asks only when that set holds more than one.
- **Corrections.** [YAY Robot](https://yay-robot.github.io/) takes spoken
  corrections mid-task ("stop", then what to do instead) and learns from them,
  improving success by up to **45%**.
- **Initiative.** [Patel and Chernova](https://arxiv.org/abs/2609.28910)
  (September 2026) define levels from reactive to unprompted. Their advice: act
  when confident, stay silent when unsure, and ask only when a mistake would be
  costly. Methods scoring above **0.6** offline collapsed to **0.002–0.296**
  when users reacted in a closed loop. In a
  [CHI 2019 study](https://dl.acm.org/doi/fullHtml/10.1145/3290605.3300328),
  highly proactive robots felt controlling and interrupting, and medium
  proactivity was rated more helpful and was preferred.
- **Not interrupting.** [NIABench](https://arxiv.org/abs/2605.01368) (May 2026)
  helps with the human's plan without breaking into it, treating that plan as
  the main process.
- **Turn-taking.** A
  [2026 industry guide](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/)
  puts the production bar at a 200–400 ms turn gap, under 2% false barge-ins,
  and a TTS flush under 60 ms. [Lu et al.](https://arxiv.org/abs/2609.13117)
  found people adapt their speech mid-turn when overlapped **68%** of the time,
  against **35%** for a full-duplex model.
- **Capabilities.** In a [120-person study](https://arxiv.org/abs/2502.01448)
  (HRI '25), people preferred a robot that said what it could do up front; they
  talked to it more naturally and enjoyed it more.
- **Transparency.** Trust grew with mechanistic transparency in
  [Show Your Work](https://dl.acm.org/doi/pdf/10.1145/3776734.3794439) (HRI
  2026). But people rarely act on passive transparency, which is why an
  explanation is better spoken than left on a panel
  ([Frontiers, 2026](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1935527/full)).

**For this runtime:** the persona's levels and its medium default match the evidence.
The gaps are barge-in **(1)**, spoken capabilities and reasons **(2)**, measured
ambiguity **(5)**, learning what you accept **(11)**, and learning from
corrections **(13)**.

### 9. Measuring it

- [PARTNR](https://arxiv.org/abs/2411.00081) (Meta) has **100,000**
  natural-language tasks across 60 houses and 5,819 objects, for a robot and a
  human working together.
- [LLM user simulators](https://arxiv.org/abs/2410.23535) reproduced human
  behaviour in embodied dialogues only moderately (F-measure around
  **42–43%**). That makes them useful for regression tests, but not a
  replacement for people.
- Patel and Chernova's closed-loop result above is the warning: judge the
  system with a user who reacts to it.

**For this runtime:** there was no regression suite, only one-off scripts. A
scenario suite with simulated users on THOR would make every other change here
measurable **(6)**.

## Roadmap

Ordered by what gives the most for the least. Numbers match the map above.

### Now · a day or two each

| # | Change | Why | Where | Effort | Now |
|---|---|---|---|---|---|
| 1 | When you start talking, the robot pauses its speech, then continues, adapts or stops depending on what you said. | Target 200–400 ms turn gap; people adapt mid-turn 68% of the time, models 35% | `ui/server.py`, speech queue in `agent/skills.py` | S | Dropped: voice was removed |
| 2 | Say up front what it can and can't do (it can't open the microwave yet), and give a reason when plans change ("not on the counter, checking the shelf"). | Proactive capability disclosure preferred (HRI '25); passive transparency rarely used | `agent/persona.py`, `agent/model.py` | S | Partly: the chat's progress lines give the reason when a plan changes; no up-front capability statement yet |
| 3 | Search by expected cost: the planner rates each spot once, and code orders the visits by likelihood against travel cost, then verifies and accepts, continues or gives up. | Up to 11.8% cheaper than the LLM choosing; Finder +15.75 points | `agent/harness.py`, a new search goal | S–M | Not started |
| 4 | Prompt diet: only nearby and request-relevant objects, plus a `recall(query)` tool over memory and notes. | AgenticNav's recall tool; LT-Mem 10× fewer tokens; the runtime was at about 6k tokens a call | `agent/model.py`, `brains/interface.py` | S–M | Done: `recall` tool and a trimmed prompt, about 4,200 tokens a decision |
| 5 | Measure ambiguity: collect the plausible targets (two mugs for "my mug"), and ask only when there is more than one. | KnowNo's guarantee; ask only when a mistake is costly | `agent/model.py`, `agent/harness.py` | M | Not started |

### Next · a week or two each

| # | Change | Why | Where | Effort | Now |
|---|---|---|---|---|---|
| 6 | Scenario suite on THOR with simulated users; score success, time, model calls, tokens, unasked actions and questions asked. | PARTNR; closed-loop scores differ wildly from offline ones | `tests/`, a new runner | M | Done: `eval/suite.py`, ten scripted scenarios against the live server |
| 7 | Volatility-aware memory: hold a stable object after one missed look, overwrite things that move, and keep several hypotheses when unsure. Add containers and rooms. | LT-Mem; SayPlan and HOV-SG hierarchies | `agent/memory.py`, `agent/state.py` | M | Mostly: history, usual place, volatility and rooms; no containers or multiple hypotheses |
| 8 | Write the runtime rules as temporal contracts over the action trace, checked live and on saved runs. Add room-specific safety rules that a clever request can't talk its way past. | ContrAgent; governance layer 96.2% interception; RoboGuard; RoboPAIR | `_check` in `agent/harness.py` → a contracts module | M | Not started |
| 9 | A visual check from the head camera (Gemini Robotics-ER pointing and success detection), graded against THOR's truth. | ER 1.5 success detection 0.79–0.80; AHA feedback +21.4% success | `thor/robot.py`, a perception adapter | M | Not started: perception is still a stand-in |
| 10 | Think while moving: ask for the next step while a move runs, and cache routine sequences. The stale check throws away anything invalidated. | AgenticCache −65% latency, −50% tokens | the think loop in `agent/harness.py` | M | Not started |
| 11 | The persona learns what you accept: count yeses, nos and interruptions, and adjust its level and thresholds per person. | Act when confident, stay quiet when unsure; medium proactivity preferred | `agent/persona.py` | M | Not started |

### Later

| # | Change | Why | Where | Effort | Now |
|---|---|---|---|---|---|
| 12 | A learned policy (VLA) for pick and place behind the same goal API, with the runtime unchanged. | Helix, GR00T N1, Gemini Robotics 1.5, Hi Robot | `thor/robot.py` → a skill server | L | Not started |
| 13 | Learn from corrections and failures: turn them into procedural rules that guide the planner next time. | YAY Robot up to +45%; Procedural Graphs | `agent/memory.py`, `agent/model.py` | L | First version: procedural graph plus rules kept only if the suite agrees (`eval/evolve.py`) |

## How far to trust these numbers

- Most figures come from paper abstracts and summaries, not full reads, and many
  2026 papers are preprints that haven't been peer reviewed.
- Results come from different tasks, robots and simulators, so they show
  direction, not how much this runtime would gain.
- The barge-in targets come from an industry guide, not a study.
- An earlier Codex chat named "OpenRAL" as a close relative of this runtime; no such
  project turned up in this search, so it isn't included.

## Sources

**Architectures**
- [Helix, Figure](https://www.figure.ai/news/helix)
- [GR00T N1, NVIDIA (2025)](https://arxiv.org/abs/2503.14734)
- [Gemini Robotics 1.5 (2025)](https://arxiv.org/abs/2510.03342)
- [Gemini Robotics ER 1.6 (April 2026)](https://deepmind.google/blog/gemini-robotics-er-1-6/)
- [Hi Robot, Physical Intelligence (2025)](https://arxiv.org/abs/2502.19417)

**Harnesses**
- [Show-Harness (September 2026)](https://arxiv.org/abs/2609.10522)
- [AgenticNav (June 2026)](https://arxiv.org/abs/2606.10577)
- [Robot-use agents, Isola (September 2026)](https://web.mit.edu/phillipi/www/writing/robot-use-agents.html)
- [ROSA, NASA JPL](https://github.com/nasa-jpl/rosa)
- [EdgeVox](https://github.com/nrl-ai/edgevox)

**Governance and safety**
- [Runtime governance for embodied agents (April 2026)](https://arxiv.org/abs/2604.07833)
- [ContrAgent, temporal contracts (September 2026)](https://arxiv.org/abs/2609.18128)
- [RoboGuard (2025)](https://arxiv.org/abs/2503.07885)
- [RoboPAIR, jailbreaking LLM robots (2024)](https://arxiv.org/abs/2410.13691)

**Memory and search**
- [SayPlan (2023)](https://arxiv.org/abs/2307.06135)
- [HOV-SG (2024)](https://arxiv.org/abs/2403.17846)
- [ConceptGraphs](https://concept-graphs.github.io/)
- [ReMEmbR, NVIDIA](https://nvidia-ai-iot.github.io/remembr/)
- [LT-Mem (August 2026)](https://arxiv.org/abs/2608.19059)
- [Procedural Graphs (September 2026)](https://arxiv.org/abs/2609.09153)
- [LLM-informed object search (IROS 2026)](https://arxiv.org/abs/2603.23800)
- [Finder (September 2026)](https://arxiv.org/abs/2609.18058)

**Verification and latency**
- [AHA, NVIDIA](https://arxiv.org/abs/2410.00371)
- [Code-as-Monitor](https://arxiv.org/abs/2412.04455)
- [AgenticCache (MLSys 2026)](https://arxiv.org/abs/2604.24039)
- [Thinking while driving](https://www.arxiv.org/pdf/2512.10610)
- [Robotouille](https://arxiv.org/abs/2502.05227)
- [Context-aware prefaces (July 2026)](https://arxiv.org/abs/2607.23204)

**Human–robot interaction**
- [KnowNo (2023)](https://arxiv.org/abs/2307.01928)
- [YAY Robot (2024)](https://yay-robot.github.io/)
- [Robots That Take Initiative, Patel and Chernova (September 2026)](https://arxiv.org/abs/2609.28910)
- [Service robot proactivity, CHI 2019](https://dl.acm.org/doi/fullHtml/10.1145/3290605.3300328)
- [NIABench, non-intrusive assistance (May 2026)](https://arxiv.org/abs/2605.01368)
- [Continue, Adapt, or Yield (September 2026)](https://arxiv.org/abs/2609.13117)
- [Barge-in guide 2026](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/)
- [What Can You Say to a Robot? (HRI '25)](https://arxiv.org/abs/2502.01448)
- [Show Your Work (HRI 2026)](https://dl.acm.org/doi/pdf/10.1145/3776734.3794439)
- [Trust calibration (Frontiers 2026)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1935527/full)

**Evaluation**
- [PARTNR, Meta (2024)](https://arxiv.org/abs/2411.00081)
- [Simulating user agents for embodied conversational AI (2024)](https://arxiv.org/abs/2410.23535)
