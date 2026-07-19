#!/usr/bin/env python3
import argparse
import json

import numpy as np


DAMPING = 1.1
A_DOUBLE_INTEGRATOR = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -DAMPING, 0.0],
        [0.0, 0.0, 0.0, -DAMPING],
    ]
)
B_DOUBLE_INTEGRATOR = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [DAMPING, 0.0],
        [0.0, DAMPING],
    ]
)


def require_scipy():
    try:
        from scipy.integrate import solve_ivp
        from scipy.linalg import solve_continuous_are
    except ImportError as exc:
        raise ImportError("lqr.py requires SciPy. Install python3-scipy or scipy in the active environment.") from exc
    return solve_continuous_are, solve_ivp


def closed_loop_alpha(a, b, k):
    eigvals = np.linalg.eigvals(a - b @ k)
    return float(max(0.0, -np.max(np.real(eigvals))))


def certified_metric_alpha(a, b, k, p):
    """Largest alpha satisfying Acl.T P + P Acl <= -2 alpha P."""
    acl = np.asarray(a, dtype=float) - np.asarray(b, dtype=float) @ np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(p)
    if np.min(eigvals) <= 0.0:
        raise ValueError("P must be positive definite")
    inverse_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    decay_matrix = -(acl.T @ p + p @ acl)
    normalized = inverse_sqrt @ decay_matrix @ inverse_sqrt
    normalized = 0.5 * (normalized + normalized.T)
    return float(max(0.0, 0.5 * np.min(np.linalg.eigvalsh(normalized))))


def solve_care(a=None, b=None, q=None, r=None):
    solve_continuous_are, _ = require_scipy()
    a = np.array(A_DOUBLE_INTEGRATOR if a is None else a, dtype=float)
    b = np.array(B_DOUBLE_INTEGRATOR if b is None else b, dtype=float)
    q = np.diag([10.0, 10.0, 1.0, 1.0]) if q is None else np.array(q, dtype=float)
    r = np.eye(b.shape[1]) if r is None else np.array(r, dtype=float)
    p = solve_continuous_are(a, b, q, r)
    k = np.linalg.solve(r, b.T @ p)
    return p, k, closed_loop_alpha(a, b, k)


def solve_tvlqr(a_seq, b_seq, q, r, qf, t_grid):
    _, solve_ivp = require_scipy()
    a_seq = np.array(a_seq, dtype=float)
    b_seq = np.array(b_seq, dtype=float)
    q = np.array(q, dtype=float)
    r = np.array(r, dtype=float)
    qf = np.array(qf, dtype=float)
    t_grid = np.array(t_grid, dtype=float)
    n = q.shape[0]

    def interpolate_matrix(seq, t):
        values = np.empty(seq.shape[1:])
        for index in np.ndindex(values.shape):
            values[index] = np.interp(t, t_grid, seq[(slice(None),) + index])
        return values

    rinv = np.linalg.inv(r)

    def rhs(t, flat_p):
        p = flat_p.reshape((n, n))
        a = interpolate_matrix(a_seq, t)
        b = interpolate_matrix(b_seq, t)
        p_dot = -(a.T @ p + p @ a - p @ b @ rinv @ b.T @ p + q)
        return p_dot.reshape(-1)

    solution = solve_ivp(rhs, (float(t_grid[-1]), float(t_grid[0])), qf.reshape(-1), t_eval=t_grid[::-1])
    if not solution.success:
        raise RuntimeError(f"Time-varying Riccati solve failed: {solution.message}")
    p_seq = solution.y.T[::-1].reshape((len(t_grid), n, n))
    k_seq = np.array([rinv @ b_seq[index].T @ p_seq[index] for index in range(len(t_grid))])
    return p_seq, k_seq


def main():
    parser = argparse.ArgumentParser(description="Solve CARE for the damped double-integrator model.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable P, K, and alpha.")
    args = parser.parse_args()
    p, k, alpha = solve_care()
    payload = {"P": p.tolist(), "K": k.tolist(), "alpha": alpha}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"P=\n{p}\nK=\n{k}\nalpha={alpha:.6f}")


if __name__ == "__main__":
    main()
