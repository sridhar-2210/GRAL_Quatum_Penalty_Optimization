"""Independent check of the NumPy QAOA simulator against Qiskit (run on the laptop).
    python tests/test_qiskit_crosscheck.py
Builds the same QAOA circuit in Qiskit (H layer, diagonal cost phase, RX mixers)
and compares the final statevector with gral.qaoa.qaoa_state up to global phase."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gral import TSPEncoding, random_euclidean  # noqa: E402
from gral.qaoa import qaoa_state, qubo_spectrum  # noqa: E402


def test_statevector_matches_qiskit():
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    from qiskit.quantum_info import Statevector
    enc = TSPEncoding(random_euclidean(4, 5))                  # 9 qubits
    W, h, off = enc.build(1.5)
    E = qubo_spectrum(W, h, off)
    C = (E - E.min()) / (E.max() - E.min())
    N = enc.N
    rng = np.random.default_rng(0)
    for p in (1, 2, 3):
        theta = rng.uniform(0, np.pi, 2 * p)
        qc = QuantumCircuit(N)
        qc.h(range(N))
        for l in range(p):
            qc.append(DiagonalGate(list(np.exp(-1j * theta[l] * C))), list(range(N)))
            for q in range(N):
                qc.rx(2 * theta[p + l], q)
        ref = Statevector(qc).data
        mine = qaoa_state(theta, C, N)
        overlap = abs(np.vdot(ref, mine))
        assert overlap > 1 - 1e-9, (p, overlap)
    print(f"  (qiskit {qiskit.__version__}: statevectors agree for p = 1, 2, 3)")


if __name__ == "__main__":
    test_statevector_matches_qiskit()
    print("PASS test_statevector_matches_qiskit")
