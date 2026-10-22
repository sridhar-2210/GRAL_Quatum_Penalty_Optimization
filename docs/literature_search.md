# Phase 0 — Systematic literature search

Goal: confirm no published method already learns per-constraint, online, transferable penalty control for QUBO / QAOA solvers. Record every hit in a table (title, year, venue, what it controls, scalar or per-constraint, online?, learned?, transfers?).

## Databases
Scopus, IEEE Xplore, ACM Digital Library, arXiv (quant-ph, cs.LG, math.OC), Google Scholar (first 100 results per string).

## Search strings
1. ("reinforcement learning" OR "policy gradient" OR PPO) AND (QUBO OR "Ising") AND (penalty OR "Lagrange multiplier")
2. (QAOA OR "quantum annealing") AND ("penalty weight" OR "penalty parameter" OR "penalty schedule") AND (adaptive OR learned OR learning)
3. ("graph neural network" OR GNN) AND ("Lagrangian multiplier" OR "penalty") AND ("combinatorial optimization")
4. "augmented Lagrangian" AND (QUBO OR "quantum annealing" OR QAOA)
5. ("learning to optimize" OR "learned optimizer") AND ("constraint" OR "penalty") AND ("quantum")

## Already identified (must be cited and compared against)
| Paper | Controls | Scalar / per-constraint | Online | Learned | Transfers |
|---|---|---|---|---|---|
| Farhi, Goldstone, Gutmann (2014) QAOA | — (foundation) | | | | |
| Yao, Bukov, Lin (2020) Policy-gradient QAOA, PMLR 107 | circuit angles | — | yes | RL | partly |
| Cheng et al. (2024) DARBO, Commun. Phys. | circuit angles | — | yes | Bayesian opt. | no |
| Ruan et al. (2023) constrained QAOA, Inf. Sci. 619 | mixer design | — | — | no | — |
| Demelas et al. (2023/24) arXiv:2310.14659 | Lagrange multipliers (MILP) | per-constraint | no (one-shot) | GNN | yes |
| Stooke et al. (2020) PID Lagrangian, ICML | multipliers (safe RL) | per-constraint | yes | no (PID rule) | — |
| Rusnáková et al. (2025/26) arXiv:2509.19005 | penalty (annealer) | scalar | no | gradient boosting | yes |
| Alessandroni et al. (2026) arXiv:2604.02416 | penalty (approx. solvers) | ? | no | no (analytic) | — |
| Ferrari, Vandelli, Dragoni (2026) arXiv:2606.25117 | penalty schedules in QAOA | per-constraint | yes (in-circuit) | variational | **no** |
| Rezk & Gora (2026) arXiv:2609.04593 | QUBO penalties (CVRPTW) | per-constraint | rule-based | no | — |

If the search finds a paper that already does all five properties, stop and re-scope with your supervisor before Phase 3.
