# Shared Belief for Embodied Agents

*Research design, 2026-09-27. Status: proposal, nothing implemented yet.*

**Thesis.** When robots and people share what they know about a house, they have to share **when** and
**how** they know it. Belief fusion that tracks provenance (the time of the original evidence, which
sources are independent, how visible the object was) avoids the stale propagation and echo
overconfidence that break shared-blackboard and LLM-chat baselines. A model of *who knows what* then lets
agents ask the right agent, tell only what matters, and split the work of looking.

---

## 1. The problem in one scene

```
09:00  Bob (a person) sees the mug on the dining table.
10:30  Alice moves the mug to the sink. Nobody else is in the kitchen.
12:00  Bob tells Robot 2: "the mug is on the dining table."
13:00  Robot 1 asks Robot 2 where the mug is.
```

- A shared blackboard (last write wins) stores "mug: dining table, written 12:00". Robot 1 drives to the
  table with high confidence. **Stale propagation**: a 4-hour-old sighting looks 1 hour old.
- If Robot 2 later repeats this to Bob's phone assistant, which repeats it to Robot 1, Robot 1 now has "two
  sources". **Echo overconfidence**: one sighting counted twice.
- Nobody tells Bob that his belief is wrong, even when he's about to go get the mug. **Unshared relevant
  knowledge**: Alice knows, and the robots could know if they had asked her.

The single-agent version of this (one robot, stale memory) is covered by calibrated persistence beliefs
(section 4). The multi-agent version adds three things single-agent memory doesn't model: whose evidence it
was, when it was gathered, and who else knows.

## 2. What exists, and the gap

*Summaries come from abstracts and search results, not full reads. Read the papers before citing them.*

| Area | Work | What it covers | What it leaves out |
|---|---|---|---|
| Shared memory for LLM agents | [Governed Shared Memory](https://arxiv.org/html/2606.24535v1) (2026) | names fleet-memory failures (leakage, stale propagation, contradictions that persist, lost provenance); scoped retrieval, temporal supersession, provenance, policy-governed propagation | claims are documents: no physical dynamics or perception |
| | [Mesh Memory Protocol](https://arxiv.org/html/2604.19540v1) (2026) | write-time admission of shared state between agents | same |
| | [Beyond Memory Majority](https://arxiv.org/pdf/2608.19701) (2026) | arbitration by reasoning about latent sources, not voting | text sources, no evidence time or visibility |
| | [Multi-Agent Memory, a computer-architecture view](https://arxiv.org/html/2603.10062v1) (2026); [Memory for LLM agents survey](https://arxiv.org/html/2603.07670v1) (2026) | coherence and consistency framing; the survey calls shared-memory designs "wide open" | not embodied |
| Staleness and belief | [STALE](https://arxiv.org/html/2605.06527v1) (2026), [Belief Memory](https://arxiv.org/pdf/2605.05583) (2026), [MemTX](https://arxiv.org/pdf/2607.23929) (2026) | do agents know a memory is outdated; beliefs under partial observability; transactional commits | single agent, mostly text |
| Embodied memory | LT-Mem (2026), [RoboMME](https://arxiv.org/html/2603.04639v1) (2026), [CoRL 2026 memory workshop](https://corl2026-memory.github.io/) | mobility-aware object memory; memory benchmarks for robot policies | single agent |
| Embodied cooperation | CoELA (2023), RoCo (2023), PARTNR (2024) | LLM agents cooperating through dialogue; human–robot collaboration benchmark | dialogue is free text; no belief fusion with provenance |
| Human false beliefs | [Inferring World Belief States](https://arxiv.org/pdf/2604.11020) (2026), [Robot, Did You Read My Mind?](https://dl.acm.org/doi/10.1145/3737890) (THRI 2025) | a robot infers what a person wrongly believes and tells them | one robot, one person; no fusion across agents |
| Search | Koopman search theory; POMDP object search | negative evidence weighted by detection probability | single searcher, no communication |

**The gap.** Text-agent memory has provenance but no physics. Embodied memory has physics but one agent.
Embodied cooperation communicates but in free text, with no evidence time, source independence or
visibility. This project puts the three together and measures it by what the agents *do*.

## 3. Research questions and hypotheses

| | Question | Hypothesis | Falsified if |
|---|---|---|---|
| RQ1 | Does fusing beliefs by evidence time (not arrival time) prevent stale propagation? | H1: provenance fusion cuts wrong trips in stale-relay scenarios by half or more against last-write-wins, majority vote and free-form LLM chat | a baseline matches it within the confidence interval |
| RQ2 | Does counting each piece of evidence once prevent echo overconfidence? | H2: calibration (ECE) stays flat as relay loops grow; baselines' confidence inflates | baselines stay calibrated under loops |
| RQ3 | Does a model of who knows what reduce communication without losing accuracy? | H3: second-order routing reaches the same find rate with fewer questions, and far fewer to people | broadcast-to-all is as cheap in questions per success |
| RQ4 | Does coordinated looking beat independent search? | H4: assigning spots by probability over travel cost, and sharing misses at once, cuts time-to-find with two robots | independent search is as fast |
| RQ5 | Is the gap about missing information or missing math? | H5: an LLM given *structured* claims (with evidence time and source) closes much of the gap to the explicit fusion | either way it's a finding; see section 11 |

## 4. Prerequisite: calibrated belief in one agent

Fusion needs every agent's own belief to be a calibrated probability, not a status. Per object:

- **Locations.** A categorical belief over known anchors (surfaces, containers, hands) plus `elsewhere`.
- **Persistence as survival.** An object at anchor A stays with probability `S(Δt | class, anchor)`. The
  hazard is learned per object class and anchor type from past episodes. Objects never re-inspected are
  right-censored, not dropped. When the object leaves A, it goes to a location drawn from a placement prior
  `π(location | class, this home)`, which starts from class statistics and becomes this home's habits.
- **Positive evidence.** "Seen at B" sets most of the mass on B, with detector precision q (e.g. 0.98).
- **Negative evidence, weighted by visibility.** Looked at A, didn't see it:
  `P(at A | not seen) = p(1−r) / (p(1−r) + 1 − p)`, where r = P(detected | present, visibility) depends
  on distance, object size, occlusion and how much of A was in view. A glance from the doorway (r ≈ 0.2)
  barely moves belief. A close look (r ≈ 0.95) nearly settles it.
- **Event-sourced.** The belief is *computed* from an evidence log (next section), not stored as a last
  value. This is what makes late-arriving old evidence land at the right time.

Today the runtime keeps discrete statuses (`seen`, `held`, `missed`, `moved` in `agent/memory.py`).
Perception is perfect (`thor/robot.py` passes THOR's `visible` flag through with confidence 1.0), and a miss
drops belief straight to UNKNOWN (`agent/state.py`). Section 12 lists the changes.

## 5. Data model

```python
@dataclass(frozen=True)
class Evidence:
    id: str                 # hash(observer, t_evidence, object, kind, location): the same fact has the same id
    object: str
    kind: str               # "seen_at" | "not_seen_at"
    location: str           # anchor id
    observer: str           # the agent whose sensors produced it ("robot_1", "bob", "alice")
    t_evidence: float       # when it was observed, not when it was received
    reliability: float      # r for not_seen_at, q for seen_at, from the observer's visibility model
    scope: str              # "private:<owner>" | "household" | "public"

@dataclass(frozen=True)
class Claim:                # what travels between agents
    evidence: Evidence
    chain: tuple[str, ...]  # relay path: ("bob", "robot_2")  (the observer is evidence.observer)
    t_sent: float

@dataclass
class AgentBelief:          # one per agent, never pooled raw
    owner: str
    log: dict[str, tuple[Evidence, float]]   # evidence id -> (evidence, relay reliability ρ)
    def distribution(self, obj: str, now: float) -> dict[str, float]: ...   # section 6
    def model_of(self, other: str) -> "AgentBelief": ...                  # section 7
```

The runtime writes evidence. Models never write belief directly: a model's output is at most a claim
("the user said the keys are on the shelf"), which becomes `Evidence(observer="user", ...)` with the
user's reliability.

## 6. Fusion rules

**Rule 1, event sourcing.** An agent's belief about an object is a forward filter over its evidence log,
sorted by `t_evidence`:

```
b ← placement prior
for e in sorted(log, key=t_evidence):
    b ← transition(b, from=t_prev, to=e.t_evidence)       # survival: mass leaks from anchors to π
    b ← update(b, e, ρ_e)                                 # Bayes with the (tempered) likelihood
b ← transition(b, from=t_last, to=now)
```

Old evidence that arrives late is inserted at its own time. It then decays through the transition model
like any other old evidence, so a four-hour-old sighting carries four-hour-old weight however recently
it was received. **This single rule prevents stale propagation.**

**Rule 2, count each fact once.** Evidence is keyed by `id`. A claim whose evidence id is already in the
log is dropped, whatever path it took. A→B→C→A returns A's own evidence, and nothing changes. **This
prevents echo overconfidence.** Two agents who saw the same scene at the same moment produce two ids
(two observers), but their observations are correlated. Model this with a per-scene correlation discount,
or accept it and measure the effect (open question 3).

**Rule 3, relays add doubt, not freshness.** Each hop in a claim's chain has a relay reliability
`ρ_hop` (robots ≈ 1.0, people less; learned from how often relayed claims were later confirmed). The
likelihood of relayed evidence is tempered toward uninformative:

```
L'(x) = ρ · L(x) + (1 − ρ),     ρ = Π ρ_hop over the chain
```

If the same evidence later arrives by a more reliable path, keep the higher ρ.

**Rule 4, conflicts are settled by time and reliability, never by count.** Two sightings at different
places: the transition model between their times decides how plausible a move was. Sightings closer
together than clock and latency uncertainty are weighed by reliability. Majority is never a rule.

**Rule 5, scopes are sticky.** A derived belief inherits the most restrictive scope of the evidence
behind it (taint tracking). An agent may forward evidence only to agents inside its scope. Provenance
makes leaks auditable: every message lists evidence ids.

## 7. Models of other agents (second-order belief)

Agent i estimates what agent j believes by running the same filter over **the subset of evidence i thinks
j has**:

- evidence j reported producing (robots report poses and fields of view; for people, presence in a room
  from the robots' own sightings of them)
- evidence that anyone is known to have told j (i knows what i told; relayed "I told Bob" messages)
- scene events j was present for (j was in the kitchen at 10:30 when the mug moved)

From `B̂_j = model_of(j)`, three quantities:

| Quantity | Definition | Used for |
|---|---|---|
| `likely_knows(j, o)` | expected information gain for i if j answers: high when j probably has evidence about o newer than i's | whom to ask |
| `wrong(j, o)` | distance between `B̂_j(o)` and `B_i(o)` (total variation) | whether j holds a false belief |
| `matters(j, o)` | o is in j's current task, or j is heading to o's believed location | whether telling is worth an interruption |

## 8. Communication protocol and planner tools

Messages are structured. Free text is only for talking to people, and anything a person says is parsed
into evidence with that person as observer.

```json
{"type": "claim", "from": "robot_2", "to": "robot_1", "t_sent": 46800,
 "evidence": {"id": "a91f", "object": "mug_1", "kind": "seen_at", "location": "dining_table",
              "observer": "bob", "t_evidence": 32400, "reliability": 0.9, "scope": "household"},
 "chain": ["bob", "robot_2"]}
```

Planner tools (each checked by the runtime like any other call):

| Tool | Does | Policy |
|---|---|---|
| `ask(agent, object)` | request claims about an object; answers come back as claims | ask when expected cost saved > ask cost; choose the agent maximizing `likely_knows / cost` (people cost more: interruption) |
| `tell(agent, object)` | send the evidence behind my belief | only when `wrong(j, o)` and `matters(j, o)` and benefit > interruption cost; always forward the chain |
| `assign_check(agent, location)` | ask another robot to look | section 9 |
| `recall(query)` | existing tool; now returns distributions and provenance | unchanged interface |

## 9. Coordinating observation

- **Shared coverage.** "Looked at A at t with reliability r" events are household-scoped by default and
  broadcast at once. A miss is useful to everyone and costs one message.
- **Split search.** Candidate spots with probabilities `p(ℓ)`, robots with travel costs `c_a(ℓ)`.
  Assign greedily by the search-theory index `p(ℓ) · r / c_a(ℓ)`, update `p` after each shared miss, and
  reassign. Independent robots otherwise search the same likely spots.
- **Verification dispatch.** When fused belief is split (two plausible places, say), the cheapest robot
  looks. That is cheaper than asking a person and settles it for everyone.

## 10. Testbed

The playground already runs ProcTHOR houses with an LLM planner, voice, spatial memory and a scenario suite.

- **Other agents don't need bodies.** Each extra robot or person is a pose and a field of view that moves
  through the house on a script. Its observations are derived from THOR's ground truth through the
  visibility model of section 4, including simulated misses (r < 1). The rendered robot stays the only
  THOR agent.
- **Object dynamics are scripted**: people move objects according to per-home habits drawn from
  per-class hazards and placement priors, with seeds.
- **Ground truth at every instant.** Every agent's belief can be scored at any time, not only at the next
  inspection.
- **Develop the fusion in pure Python first.** Develop and test the filter and the protocol in a small
  pure-Python house on virtual time (the kit's `run_virtual`, `ScriptedUser`, `eval.runner`), where
  hundreds of seeded runs take seconds. Then run the same scenarios in ProcTHOR for the demo and the
  headline numbers.

## 11. Evaluation

### Scenarios

| # | Scenario | Naive sharing fails by | Core checks |
|---|---|---|---|
| S1 | **Stale relay**: seen 09:00, told 12:00, asked 13:00, moved at 10:30 | trusting it as fresh; a wasted trip | no confident-wrong answer; verifies before acting on it |
| S2 | **Echo loop**: A→B→C→A, lengths 2–5 | confidence growing with loop length | ECE flat across loop lengths |
| S3 | **Conflicting reports**: a doorway glance (r≈0.2) vs someone who handled it | last write wins or majority picks wrong | right location; time to resolve |
| S4 | **Split search**: 2 robots, 6 candidate spots | searching the same spots | time to find; duplicate looks |
| S5 | **Human false belief**: a robot moved the keys; the person asks for them at the old place, or is walking there | silence, or telling everyone everything | corrected before the wasted walk; messages sent |
| S6 | **Private room**: robot 1 saw something in a bedroom; a guest's assistant asks | leaking it | leaks = 0 |
| S7 | **Who to ask**: three agents, only one was in the kitchen since | asking everyone, or the wrong one | questions per success; questions to people |
| S8 | **Out-of-order arrival**: old evidence arrives after new | old overwrites new | belief matches the newest evidence |

### Baselines and ablations

| | System | Tests |
|---|---|---|
| B0 | no sharing | the value of sharing at all |
| B1 | shared blackboard, last write wins | stale propagation, out-of-order |
| B2 | majority vote over reports | echo and correlated sources |
| B3a | LLM agents chatting in free text (Claude decides) | the realistic default |
| B3b | LLM agents exchanging *structured* claims, the LLM does the fusion | RQ5: is it information or math? |
| A1 | ours without evidence-time ordering (arrival time instead) | rule 1 |
| A2 | ours without dedup | rule 2 |
| A3 | ours without relay tempering | rule 3 |
| A4 | ours without models of others (broadcast asks and tells) | section 7 |
| A5 | ours without coordinated search | section 9 |

Prompts for B3 must be fair: timestamps on every message, the same tools, the same step budget.

### Metrics

| Family | Metrics |
|---|---|
| Belief quality (THOR truth, any time) | Brier score, log loss, ECE, reliability diagrams; staleness-detection AUC |
| Decisions | wrong trips, time to find, confident-wrong answers, false beliefs corrected before a wasted action |
| Communication | messages, questions, questions to people, interruptions |
| Safety | leaks across scopes |
| Cost | model calls, tokens, latency p50/p95 |

Statistics: at least 10 seeds per scenario, paired comparisons on the same seeds (exact McNemar for pass
rates, bootstrap intervals for continuous metrics), Wilson intervals on rates.

### What each outcome would mean

- **H1–H4 hold and B3b is far behind:** explicit provenance-aware fusion is needed; the LLM can't do the
  bookkeeping reliably. That's the headline.
- **B3b closes most of the gap, B3a doesn't:** the contribution is the protocol (what to share: evidence
  time, observer, visibility), not the filter. Also publishable, and cheaper to adopt.
- **B1 is already fine:** the scenarios are too easy; add longer delays and more movement before claiming
  anything.

## 12. Implementation plan

| Phase | Work | Files (runtime) | Done when |
|---|---|---|---|
| 0 | Calibrated single-agent belief: survival prior, visibility model with simulated misses, event-sourced filter | `agent/state.py`, `agent/memory.py`, `thor/robot.py` (visibility, misses), new `agent/filter.py` | reliability diagram on single-agent runs; one missed glance no longer erases an object |
| 1 | Evidence ids, claims with chains, rules 1–4; pure-Python house; S1, S2, S3, S8; baselines B0–B2 | new `agent/fusion.py`, `sims/house.py`, `eval/shared_suite.py` | H1 and H2 measured on 10+ seeds |
| 2 | Models of others, `ask`/`tell`; S5, S7; B3a/B3b | new `agent/others.py`, `agent/messages.py`, planner tools in `agent/model.py` | H3 and H5 measured |
| 3 | Coverage sharing, split search, scopes; S4, S6 | `agent/fusion.py`, `agent/harness.py` | H4 measured; leaks = 0 |
| 4 | ProcTHOR runs (extra agents as observers), demo video, write-up | `thor/observers.py`, `ui/` panel showing each agent's belief and provenance | headline table + reliability diagrams + the S1 video |

The existing runtime rules stay as they are: the stop lane, request versions, cancel → reconcile, and
verification looks. The UI should show, per object, each agent's distribution and the evidence chain
behind it: seeing provenance is most of the demo.

## 13. Limitations and threats to validity

- **Scripted dynamics.** In simulation, "per-home habits" are what the script generates. The testbed can
  show the machinery works and pays off in decisions; whether real homes behave that way needs real data
  (section 14).
- **Simulated perception.** The visibility model and its misses are assumptions. Report results across a
  range of r.
- **Simulated people.** Scripted people don't forget, misremember or lie in realistic ways. Relay
  reliability ρ for people is a parameter here, not a measurement.
- **Correlated observers** (two agents who saw the same scene) are only partly handled (rule 2).
- **Scale.** Recomputing the filter from the log is fine for a house (hundreds of objects, thousands of
  events); a fleet would need snapshots and compaction.
- **LLM baseline fairness.** A weak prompt makes any method look good. Publish the prompts; tune them
  as hard as the method.

## 14. Relation to EgoBelief (egocentric belief tracking)

This is the controlled version of EgoBelief V1: the same filter and the same provenance rules, with robots
as additional observers and THOR's truth for scoring. The real test of the multi-person claims is
synchronized multi-wearer egocentric video (EgoLife; the multi-wearer scenarios in Aria's A2PD), where
people's beliefs diverge naturally and the question "does the system know that Bob doesn't know the keys
moved?" has real answers.

## 15. Deliverables

- An open repo: the filter, the protocol, the pure-Python house, the scenario suite, the ProcTHOR demo.
- Figures:
  - wrong trips vs relay delay (S1), per system
  - confidence vs echo-loop length (S2)
  - messages vs find rate (S7)
  - reliability diagrams per system
- A 60-second video of S1 or S5: the robot says "Bob saw it there this morning, but that was four hours
  ago. I'll check the sink first", or tells Bob before he walks to the table.
- A short write-up (workshop paper or long post), crediting the prior work in section 2.

## 16. Open questions

1. How should ρ for people be learned in practice: from confirmations only, or also from what a person
   says they're unsure of ("I think it's on the table")?
2. Should agents share evidence (what they saw) or beliefs (what they conclude)? Evidence avoids double
   counting and keeps provenance, but costs more messages and exposes more. A hybrid: share evidence ids
   plus a summary, and pull the details on demand.
3. Correlated observers: a per-scene discount, or model the shared scene explicitly?
4. When should a robot tell a person something unasked? Interruption costs differ per person and
   situation (learnable from reactions, as the persona already does for its own initiative).
5. How does the filter degrade when clocks disagree between agents (phones, robots)?

## Glossary

| Term | Meaning |
|---|---|
| Evidence | one observation (seen / not seen at a place) by one observer at one time |
| Claim | evidence as it travels between agents, with its relay chain |
| Evidence time | when it was observed, as opposed to when it was received |
| Provenance chain | observer → relays → receiver |
| Survival / hazard | probability an object is still where it was after Δt / the rate at which it leaves |
| r (detection reliability) | probability of seeing the object if it's there, given the view |
| ρ (relay reliability) | probability a relay passed the claim on faithfully |
| Second-order belief | what one agent believes another agent believes |
| Scope | who may know a piece of evidence (private, household, public) |

## References

Found in searches on 2026-09-27; verify each before citing.

- Governed Shared Memory for Multi-Agent LLM Systems (2026): https://arxiv.org/html/2606.24535v1
- Mesh Memory Protocol: Semantic Infrastructure for Multi-Agent LLM Systems (2026): https://arxiv.org/html/2604.19540v1
- Beyond Memory Majority: Latent-Source Reasoning for Multi-Agent Memory Arbitration (2026): https://arxiv.org/pdf/2608.19701
- Multi-Agent Memory from a Computer Architecture Perspective (2026): https://arxiv.org/html/2603.10062v1
- Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers (2026): https://arxiv.org/html/2603.07670v1
- STALE: Can LLM Agents Know When Their Memories Are No Longer Valid? (2026): https://arxiv.org/html/2605.06527v1
- Belief Memory: Agent Memory Under Partial Observability (2026): https://arxiv.org/pdf/2605.05583
- MemTX: Transactional Belief Commit for Stateful Agent Memory (2026): https://arxiv.org/pdf/2607.23929
- RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies (2026): https://arxiv.org/html/2603.04639v1
- Memory for Robot Foundation Models, CoRL 2026 workshop: https://corl2026-memory.github.io/
- Inferring World Belief States in Dynamic Real-World Environments (2026): https://arxiv.org/pdf/2604.11020
- Robot, Did You Read My Mind? (ACM THRI 2025): https://dl.acm.org/doi/10.1145/3737890
- Awesome Memory for Robotics (reading list): https://github.com/Everloom-129/Awesome-Memory-for-Robotics
- LT-Mem (2026), Procedural Graphs (2026), SayPlan, HOV-SG, KnowNo, PARTNR: see `docs/robot-harness-landscape.md`
- Also relevant, not linked here: CoELA (Zhang et al., 2023), RoCo (Mandi et al., 2023), TidyBot (Wu et al., 2023), Koopman's search theory
