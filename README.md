# GRAL — Graph-RL Augmented-Lagrangian Penalty Control

**Honours project:** *Design and Analysis of Hybrid Classical–Quantum Algorithms for Combinatorial Optimization*

**Method name (for the paper):** *GRAL: Learning Per-Constraint Penalty Control for Hybrid Classical–Quantum Combinatorial Optimization*

---

## 1. The research gap (what makes this novel)

Constrained problems such as the TSP are turned into a QUBO by adding penalty terms. How strongly to penalise each constraint decides whether a classical annealer or a quantum circuit (QAOA) returns valid, good solutions. Existing work handles this in one of five ways, and each has a limitation:

| Existing approach | Example | Limitation |
|---|---|---|
| One fixed λ, hand- or grid-tuned | most QUBO papers | ignores solver feedback; one value for all constraints |
| Penalty predicted once before solving | Rusnáková et al. (ML, minimum bisection, 2025) | single scalar; no adaptation during solving |
| Penalty computed analytically | Alessandroni et al. (2026) | not learned; fixed before solving |
| Penalty schedules inside QAOA | Ferrari et al. (2026) | re-optimised from scratch for every instance; QAOA only |
| Hand-written adaptive rules | Rezk & Gora (2026); classical ALM | fixed formulas, not learned |

**Gap:** no published method *learns* a policy that sets **a separate penalty for every constraint**, **adapts it online from solver feedback**, and **transfers to unseen and larger instances without retraining**, across **both classical and quantum** solvers.

> A targeted search, not yet a systematic review. Before submitting, run the systematic search in `docs/` (Phase 0 below).

## 2. The method

```
TSP instance ─► per-constraint augmented-Lagrangian QUBO
                E(x) = D(x) + Σ_c [ w_c g_c(x)² + μ_c g_c(x) ]      (one w_c, μ_c per constraint)
                       │
                       ▼
        solver round (simulated annealing  or  QAOA circuit + classical angle optimiser)
                       │  samples
                       ▼
        feedback per constraint: violation frequency, residual, current w_c, μ_c
        + global: feasible fraction, best tour so far, budget used
                       │
                       ▼
        GNN policy on the constraint graph (shared weights → any problem size)
        trained with PPO → outputs Δw_c, Δμ_c for every constraint (bounded)
                       │
                       └──► next solver round
```

**Version 2 design (current):**
* **Residual learning** — GRAL learns *corrections* on top of the hand-designed per-constraint
  update rule: w_c ← clip( w_c · rule_factor_c · exp(0.5·a_c) ), μ_c ← μ_c + 0.5·a'_c.
  Action 0 reproduces the rule exactly (verified by a test), so GRAL is a *learned
  generalisation* of the classical update and starts from the strongest baseline.
* **Paired (common-random-numbers) reward** — every training episode is replayed with the
  rule on the same instance and the same solver seeds; the reward is the improvement over
  the rule. This removes instance difficulty and most solver noise from the learning signal.
* Version 1 (no residual, absolute reward) is kept in `models/v1/` as an **ablation**.

Why it is not "just PPO changing λ":
1. **Per-constraint** control instead of one scalar.
2. **Graph neural network** policy with shared weights, so one trained model works for any number of cities.
3. **Learned** update rule replacing classical dual ascent.
4. **Online**: reacts to what the solver actually returns.
5. **Solver-agnostic**: the same policy drives simulated annealing and QAOA.

## 3. Roadmap and status

| Phase | What | Status |
|---|---|---|
| 0 | Proposal approved by supervisor; systematic literature search | ☐ **do this now** |
| 1 | Encoding, exact solver, SA solver, all baselines, benchmark protocol | ✅ done (`gral/`, `notebooks/01_foundation.ipynb`) |
| 2 | RL environment + **scalar PPO** (learned baseline / ablation) | ✅ code done — run `scripts/run_phase2.py` |
| 3 | **GRAL**: GNN policy over constraint graph, PPO training, size transfer | ✅ code done — same command |
| 4 | **QAOA** solver (statevector, 9 & 16 qubits) plugged into the same loop | ✅ code done — `scripts/run_phase3.py` |
| 5 | Full experiments: 30 runs/config, held-out instances, Burma14, ablations, statistics | ☐ |
| 6 | Write-up: thesis chapters + paper draft | ☐ |

## 4. How to run (your laptop)

```
cd C:\gral_project
python scripts\run_phase3.py
```
Leave it running (do not close the window or let the laptop sleep). Send `results\phase3_log.txt` when it says ALL DONE.

### Google Colab (alternative)

1. Upload `gral_project.zip` to Colab (Files panel) or open `notebooks/01_foundation.ipynb` and follow cell 1.
2. Run all cells. Tests must print `PASS` for every check before trusting any result.
3. Results land in `results/`.

Locally: `python tests/test_core.py` then `python scripts/benchmark_baselines.py --sizes 10 12 14 --count 10 --seeds 3 --burma`.

## 5. Code map

| File | Purpose |
|---|---|
| `gral/tsp.py` | random instances, Burma14 (TSPLIB GEO), Held-Karp exact optimum |
| `gral/qubo.py` | per-constraint augmented-Lagrangian QUBO, decode/encode, constraint graph |
| `gral/sa.py` | simulated annealing (numba fast path, numpy fallback) |
| `gral/runner.py` | budgeted outer loop; `RoundStats` = what controllers observe |
| `gral/controllers.py` | baselines: fixed, grid-tuned, scalar-adaptive, per-constraint rule, ALM dual ascent |
| `scripts/benchmark_baselines.py` | fair benchmark: tune on TRAIN split, evaluate on TEST split |
| `gral/rl/features.py` | observations, bounded actions, telescoping reward (NumPy) |
| `gral/rl/env.py` | penalty-control environment (same mechanics as `run_episode`) |
| `gral/rl/policy.py` | **GRAL GNN policy** and scalar policy (PyTorch) |
| `gral/rl/ppo.py` | PPO with parallel rollouts, validation, best-checkpoint saving |
| `gral/rl/controller.py` | trained policy as a controller for fair evaluation |
| `scripts/train.py` | `--mode gral` or `--mode scalar` |
| `scripts/evaluate.py` | full benchmark incl. generalisation sizes, Burma14, Wilcoxon paired stats |
| `scripts/run_phase2.py` | one command: tests → train GRAL → train scalar PPO → evaluate |
| `gral/qaoa.py` | **QAOA solver**: exact statevector, COBYLA angle optimisation, warm start, shot sampling |
| `scripts/run_phase3.py` | **overnight run**: tests → train v3 → validation-based model selection → final classical + QAOA evaluation |
| `tests/test_qaoa.py`, `tests/test_qiskit_crosscheck.py` | QAOA vs exact matrix exponentials; vs Qiskit statevector |
| `tests/test_core.py` | correctness checks (QUBO = tour length, ground state = optimum, Burma14 = 3323) |
| `tests/test_env.py`, `tests/test_rl.py` | environment and PyTorch/PPO checks |

## 6. Experimental rules (keep these for every phase)

* Same budget (rounds × reads × sweeps) and same seeds for every method.
* TRAIN instances for tuning/training, TEST instances only for final numbers — never mix.
* Report: feasibility rate, optimality gap vs exact optimum, round of first feasible tour, runtime; mean ± std over runs.
* Count tuning/training cost separately from solving cost.
