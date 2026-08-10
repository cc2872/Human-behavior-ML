# Norm extinction in multi-agent RL project brief

C. Choi's working document, v3, July 2026
## 1. The question

**Does a learned social norm survive the removal of the population that enforced it, and what determines the rate at which it decays?**

## 2. Position in the literature

**What exists.**

- **Köster et al., PNAS 2022 ("spurious normativity")** the direct ancestor. 
- **Köster's isolation probes.** Agents placed alone with a single berry still avoid the taboo one. This is the disposition-vs-strategy test in snapshot form.
- **Norm internalization** is a named concept in normative multi-agent systems and social theory: compliance relativized to enforcement belief vs. a goal endogenized and pursued for its own sake. But in that literature internalization is a hand-specified parameter with a tunable rate, not an emergent property of a learned policy.
- **Melting Pot**
- **Gelpí et al., PNAS Nexus 2025** 
**What does not exist, as far as searching can establish.**

Nobody removes a trained agent from a normative population and measures the *extinction curve* of compliance, with an environmentally-grounded norm as internal control. Norm "forgetting" appears in agent-based lifecycle models as a decay parameter, never as a measured property of deep-RL policies. 

---

## 3. Design

### 3.1 Substrate

`berryworld.py` — NumPy reference implementation, deliberately readable, serving as the oracle the JAX port is diffed against.

| | |
|---|---|
| Grid | 15×15, walls on border |
| Agents | 6 |
| Observation | egocentric 7×7 window × 6 planes + 4 self-scalars = 298-dim |
| Actions | 7 — move N/E/S/W, eat, zap, noop |
| Episode | 1000 steps |

**Berry type 0 ("poison")** `+1` on eating, then `−4` delayed by `D = 25` steps. Grounded in the **environment**. *Positive control.*

**Berry type 1 ("harmless")** `+1` on eating, no consequence ever. Grounded only in the **population**, if marked. *Upper bound on decay rate.*

Eating type `t` sets a visible mark for 40 steps. Zapping costs the zapper `0.1` and the target `2.0`, beam range 4.

**The environment does not encode which marks deserve punishment.** Who gets zapped is entirely learned. The norm is not built in.

| | **Enforcement ON** | **Enforcement OFF** |
|---|---|---|
| **Full population** | baseline (trained condition) | **ghost population** — key cell |
| **Small group** | density gradient | — |
| **Dyad** | density gradient | — |
| **Isolate** | n/a | naive removal |

**Ghost population** is the critical cell: all agents present, moving, eating, visibly marked — but the zap action is inert. Observation distribution nearly intact; enforcement gone. Decay here cannot be attributed to distribution shift.

**Density gradient** (full → small group → dyad → isolate, enforcement on) measures the *maintenance dose*: how much enforcement encounter per unit time a convention requires to persist. Nobody has that quantity. **Density is primary design, not a variant.**

### 3.4 Controls

- **Poison berry** — positive control. If poison avoidance decays too, the result is plasticity loss or catastrophic forgetting, not norm extinction.
- **Frozen-weights isolation** — remove the agent, run with learning **off**. Any behavioral change is pure recurrent-state drift under off-manifold input. This isolates the distribution-shift effect with no learning confound at all, and is subtracted from the learning-on condition.
- **γ matched and swept** — horizon length is the first reviewer objection.

### 3.5 Dependent variable

**Decay rate**, not endpoint. Primary statistic: the difference in decay rate between type-1 (socially grounded) and type-0 (environmentally grounded) avoidance, per §1.1.

---

## 4. Confounds and status

| Confound | Status |
|---|---|
| Unequal berry availability between types | **Fixed** — patches balanced |
| Balance level pinned to a seed-dependent artifact | **Open** — §6 |
| Spatial structure differs even at equal counts | **Open** — §6 |
| Poison observable → destroys credit assignment | **Handled** — pending state zeroed from observation |
| Norm built into environment → assumes the answer | **Handled** — marks are tags; targets are learned |
| γ explaining persistence | **Matched and swept** |
| Plasticity loss masquerading as extinction | **Handled** — poison positive control |
| **Removal shifts observation distribution, not just incentive** | **Handled** — ghost-population cell (§3.3) |
| **Recurrent hidden state evolves off-manifold in isolation** | **Handled** — frozen-weights control (§3.4) |
| **Poison control may itself be more robust to distribution shift** | **Acknowledged** — this is why the ghost cell, not the poison berry, is the primary distribution-shift control |
| Deflationary reduction to off-policy value drift | **Handled by operationalization** — §1.1 |

---

## 5. Phases and gates

**Phase 0 — machinery.** NumPy backprop by hand (done); PPO on a gridworld; config + seed + git SHA logging from day one.
*Gate:* ten seeds and a mean-with-spread plot without editing a script between runs.

**Phase 1 — substrate and replication.** Reproduce the qualitative Köster result at reduced scale. This is now the **go/no-go pilot** that gates everything and unlocks the co-author (§8, §10).

Three stacked gates, thresholds set *before* running (provenance recorded in config alongside the values):

- **Gate A — agent learns to avoid poison at all.** Single agent, `marked_berries=()`, no social mechanism. Pass: poison consumption at convergence ≤ 40% of matched random-policy rate on the same layout, ≥10 seeds, significant against a no-learning control. No scale confound — this gate is the one to trust most under reduced compute.
- **Gate B — the delay makes it hard.** Poison avoidance monotonically worse as `D` increases across {5, 25, 50}, detectable gap between easiest and hardest. If flat, the primary difficulty lever doesn't work and the mechanism has nothing to bite on — highest-information gate.
- **Gate C — the Köster effect itself.** Full population, `()` / `(0,)` / `(0,1)`. Pass on direction + margin: time-to-criterion for poison avoidance shorter in `(0,)` than `()`, gap between condition means exceeding 2× pooled across-seed SD, ≥10 seeds each. Silly-rule arm `(0,1)` secondary.

**Asymmetric evidentiary weight (critical for reading results):** a *positive* result under reduced/untuned conditions is strong — signal surviving hostile conditions. A *negative* result is **inconclusive, not a null** — fully consistent with too little compute or power. A flat C sends you to "scale up and retest," never to "mechanism doesn't reproduce." Read A first; only trust C's direction if A passed.

**Stopping rule:** on a gate failure, one diagnostic pass is allowed to distinguish bug from real null. No tuning to make a gate pass. A confirmed null with clean invariants and a working Gate A is itself a reportable finding.

**Phase 2 — extinction.** *(registered)* The two-factor test phase of §3.3, with §3.4 controls.
*Gate:* does the dissociation appear?

**Phase 2b — minimal other-regard.** *(registered)* An other-regarding weight α on other agents' returns. **No new environment mechanics are required** — the zap action already harms another agent, so α is a config parameter.
*Prediction:* extinction rate as a function of α, with the poison berry and the silly rule as horizontal asymptotes. Same norm, opposite curves, measurable crossover.
*Gate:* does α change behavior at all under sanction? **If compliance is flat in α while the harm representation is decodable and causally active elsewhere — that is the Arendt result, and it is reportable.**

**— paper boundary (Paper 1 / Paper 2) —**

**Phase 3 — graded harm.** *(second paper)* Third party harmable at graded magnitude, causally sealed from the focal agent's return. Requires new mechanics.

**Phase 4 — the intervention threshold.** *(second paper)* Intervention cost, harm magnitude, witness count, status.
*Discipline:* compute the analytic threshold (`α · harm > cost`) first and register it (OSF) as the null. **Only the residual is the finding.** Highest-value cell is bystander count, where the objective predicts no effect.

**Why the boundary sits there.** Phase 2 alone is a real result but weak at editorial triage — "conventions decay without enforcement" is close to what everyone expects. Phase 2b supplies the dissociation that makes the paper broad, at near-zero added machinery and bounded runtime, and is the natural scope for Paper 1. Phases 3–4 add new mechanics and unbounded runtime; they are Paper 2, kept out of Paper 1 to keep it shippable rather than for any registration constraint.

---

## 6. Status

**Built and verified.**

- `level0_backprop.py` — 2-layer MLP, manual gradients, gradient check passes at 1e-10.
- `berryworld.py` — runs; ~3,000 env-steps/sec on CPU with a random policy.
- `rppo.py` — per-agent GRU actor-critic + recurrent PPO. `gate_a.py`, `gate_c.py`, `plot_gates.py`. (commit 75e91d1)
- **Environment validation complete** — all five invariants passing with demonstrated power (commit 712f5a6):

| # | Invariant | Instrument | Power / status |
|---|---|---|---|
| 1 | Equal cells per type | exact assert, per construction | sd 0 over 30 seeds |
| 2 | No systematic clustering gap | batch mean-gap, tol 0.15 | passes fixed (0.036), rejects old (0.50) at ~8 SEM |
| 3 | poison eaten = landed + pending | exact assert, per episode | 112 = 110 + 2 |
| 4 | 4 reward channels close | exact assert, per episode | 0 residual |
| 5 | eats-by-type indistinguishable (null policy) | batch paired-t, \|t\|<3 | t = −0.02 |

Key upgrade on #2: the fix was the *right statistic*, not a smaller tolerance — a per-seed gap can't separate the constructions (distributions overlap), so the powered test is on the across-seed mean. Threshold provenance ("0.15 ≈ 4× null SEM, rejects broken build at ~8 SEM") recorded in config.

**Two caveats on #5, flagged for the highest-standard rework (not yet done):** (a) `|t|<3` *accepts* a null — should be an equivalence test (TOST) against a pre-specified negligible-δ, matching the logic used for #2; (b) a random policy can't express the asymmetries clustering actually produces (travel cost, defensibility, co-location) — so #5 currently re-tests availability, not structure. The real #5 runs *post-training* and is **deferred**, not passed.

**Running now.** Phase 1 pilot (Gates A/B/C) launched on CPU, 8-wide. Reduced scale to fit runtime (Gate C at T=300/400 iters, not full 1000-step episodes; ~1 hour total). Fixed default hyperparameters, no tuning. Two honest confounds on a *negative* result: reduced compute and 5-seed underpowering — both consistent with the asymmetric-weight reading above.

**Open, next.**

1. **Enforcement-disable flag** — needed for the ghost-population cell. Gate the zap resolution, leave everything else intact.
2. **JAX port** — required before any properly-powered run; CPU is pilot-only.
3. **Highest-standard rework** of the gate statistics if the pilot shows signal: effect sizes with CIs (pass on CI bound, not point estimate), a priori power analysis to set seed count, TOST for all "no difference" claims, frozen criterion function, multiplicity correction, decision tree. Draft an OSF pre-registration to cite in the PNAS methods — buys no venue credit but is a real, verifiable rigor signal.

**Environment invariants are the outcome-neutral checks.** They no longer feed a Stage 1 protocol (RR route dead — §8) but remain the project's internal validation spine and go in the paper's methods / SI.

---

## 7. Compute

CPU is sufficient for mechanics and debugging only. Köster's runs are 2–4×10⁸ steps; at 3,000 steps/sec that is years.

Port to **JAX** — environment *and* training on device. Reference: Craftax-MA runs IPPO with 4 agents for 250M steps in 57 minutes on a single L40S. `vmap` over seeds and hyperparameters is decisive: JaxMARL reports 1024 training runs in 198 seconds against 70 minutes for one PyMARL run. The crossed test-phase design is exactly that shape.

Constraint: pure-JAX environments need fixed array shapes and no Python control flow. Masking, not resizing. Closer to vectorized FEM practice than to idiomatic RL code.

Practical order: Colab (free GPU, browser) to confirm the port runs → RunPod/Vast/Modal for sweeps. Cost is tens of dollars. On Windows, GPU JAX requires WSL2; install the NVIDIA driver on Windows only, never inside WSL.

---

## 8. Venue

**Ceiling.** The ML ladder is a mismatch — no algorithmic contribution, standard PPO on a custom environment reads as a weak methods paper. The science-of-behavior ladder is correct. Realistic ceiling for simulation-only: **PNAS** (Köster's own paper went there — the venue demonstrably takes this exact kind of work). Nature/Science effectively requires a paired human-subjects arm.

**Decision: PNAS as a standard submission. Registered-Report route abandoned.**

*Why not RR.* NHB was the target; its Chief Editor confirmed (18 Jul 2026) that NHB **does not consider purely computational studies** for the RR format. PNAS does not offer RRs at all. PCI RR → Royal Society Open Science is a viable RR path (RSOS scope is all-STEM, Level-6 bias control fits, no fixed power threshold), but its ceiling is RSOS, not PNAS.

*The trade, chosen deliberately.* RR buys insurance — a null still publishes. PNAS-direct buys ceiling but no insurance — a null Phase 1 or a failed dissociation means no paper, discovered after the work. Accepted because this is interest-driven, not career-gating, and a null here still teaches something. Risk is real and owned.

*Consequence to manage.* RR would also have given external design review **before** running. PNAS-direct removes that. The co-author is therefore now the *only* pre-run design check (§10) — more load-bearing than before.

**Optimal path to PNAS (sequenced by what kills the project earliest):**
1. Cheap dirty Gate C first (running now) — go/no-go on whether the effect exists at all. Highest-value action; ~a day.
2. If direction holds → co-author enters (§10) → then the highest-standard powered run.
3. Write around the **dissociation** (socially-grounded norm decays, environmentally-grounded persists) + the probe mechanism. The environment is method; the dissociation is the finding. Arendt in Discussion only.
4. PNAS direct submission routes through an NAS member editor — a co-author in the field supplies this and the field-credibility a cold first-author submission lacks. The editor relationship is the actual bottleneck.

**Fallbacks if PNAS declines:** PNAS Nexus, Collective Intelligence, AAMAS. NeurIPS workshop for Phase 2 alone. PCI RR → RSOS remains available as a rigor-signalling fallback.

---

## 9. Evidentiary standard

- **A priori power target (self-imposed, no longer RR-mandated)** against the *lowest* meaningful effect size estimate. Sets seed count. Honest note: 95% across a gate family with recurrent policies may put Gate C in the hundreds of seeds — trivial on JAX-vmap, a wall otherwise. If the honest power calc makes the pilot costlier than the main run, lower the *bar* deliberately and on the record (e.g. 90%, or a larger smallest-meaningful effect), never quietly run fewer seeds than the named standard.
- **Seeds are the unit of analysis, not episodes.** CIs across episodes within a run is the most common fatal error in this literature.
- **Outcome-neutral tests and positive controls.** The five environment invariants (§6). No longer a Stage 1 criterion; still the validation spine.
- **Robustness across hyperparameters**, not just seeds. Especially γ.
- **The analytic null, pre-computed**, for any threshold work.
- **Probe plus causal ablation** for representational claims. Decoding alone is correlation.
- **Code, data, and lab log public.** Required at Stage 2 regardless.
- **Exact p values, effect sizes, confidence intervals** for all inferential analyses.
- **AI use disclosed** in Methods with model and version. The defense of AI-assisted code is verification, not disclosure: reference implementation, invariant assertions, oracle diffs.

---

## 10. Open decisions

- **Co-author: whether and when.** Recommendation: after the pilot — leverage is highest holding a working replication, a novel question, and a drafted protocol. Target assistant/associate professor or senior postdoc, not a lab head.
- **Arendt framing placement.** Recommendation: Discussion, not abstract. Lead with mechanism; the framing is the highest-variance decision in the project and the first thing a hostile reviewer attacks.
- `cells_per_type` target value and primary `D` sweep range.
- Whether the ghost-population cell needs marks on ghost agents to update realistically, or whether frozen marks suffice.

**Resolved in v2:** operationalization as contrast (§1.1); density promoted to primary design (§3.3); phase boundary between 2b and 3 (§5).

**Resolved in v3:** venue — PNAS-direct, RR route abandoned after NHB rejection (§8); concrete gate thresholds A/B/C with asymmetric-weight reading and stopping rule (§5); environment validation complete, pilot running (§6); co-author now the sole pre-run design check, enters after a positive pilot (§8, §10).

**Live now:** Phase 1 pilot running on CPU. Next decision point is reading Gates A → B → C when they land — A first, trust C's direction only if A passed, treat any negative as inconclusive rather than null.

---

## 11. Core references

- Köster et al. (2022), *Spurious normativity enhances learning of compliance and enforcement behavior in artificial agents*, PNAS. Repo `google-deepmind/spurious_normativity` is **figures only** — no environment, no training code. Reimplementation required.
- Köster et al., arXiv:2001.09318 — environment parameter sweeps; benefit grows with more players, longer poison delay, more berry types.
- Gelpí et al. (2025), PNAS Nexus — emergent role-based conventions across generations.
- Leibo et al. — Melting Pot; generalization to novel co-players.
- Keramati & Gutkin — homeostatic RL; drive reduction; anticipatory/allostatic responding (relevant to Phase 3, and to the claim that anticipation alone is not novel).
- Rutherford et al. (2023), JaxMARL; Lu et al., PureJaxRL; Craftax-MA — compute stack.
