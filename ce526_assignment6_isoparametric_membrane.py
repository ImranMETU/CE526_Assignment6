# -*- coding: utf-8 -*-
"""
CE526 Assignment 6/7-style circular membrane problem.

Problem:
    -div(grad u) = 1 on a circular membrane of radius R = 2
    u = 0 on the outer circular boundary r = 2

Discretization:
    One 45-degree sector using:
        - one T6 quadratic triangular isoparametric element for 0 <= r <= 1
        - one Q9 biquadratic quadrilateral isoparametric element for 1 <= r <= 2

Boundary conditions:
    Outer circular boundary nodes are fixed: u = 0.
    Radial cut edges are symmetry/natural boundaries, not fixed.

The script computes:
    1. T6 and Q9 element stiffness/load vectors using Gauss quadrature
    2. Global K and F
    3. Nodal solution
    4. Radial 2D plot
    5. 3D FEM sector surface plot
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# User controls
# ------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose one:
#   "edge_midpoint" = points used in the uploaded Kaan sample:
#                     (0, 0.5), (0.5, 0), (0.5, 0.5), w = 1/6
#   "standard"      = common interior 3-point triangular rule:
#                     (1/6, 1/6), (2/3, 1/6), (1/6, 2/3), w = 1/6
TRI_RULE = "edge_midpoint"

R1 = 1.0
R2 = 2.0
THETA0 = 0.0
THETA1 = np.pi / 4.0
THETAM = 0.5 * (THETA0 + THETA1)

TENSION = 1.0
LOAD_P = 1.0


# ------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------

def exact_u(r: np.ndarray | float) -> np.ndarray | float:
    """Exact solution for -laplacian(u)=1, u(R=2)=0."""
    return (R2**2 - np.asarray(r) ** 2) / 4.0


# ------------------------------------------------------------
# Shape functions: T6 triangle
# Parent triangle: xi >= 0, eta >= 0, xi + eta <= 1
# Local order:
#   1 corner (0,0)
#   2 corner (1,0)
#   3 corner (0,1)
#   4 midside 1-2
#   5 midside 2-3
#   6 midside 3-1
# ------------------------------------------------------------

def t6_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return N, dN/dxi, dN/deta for the 6-node quadratic triangle."""
    N = np.array(
        [
            (1.0 - xi - eta) * (1.0 - 2.0 * xi - 2.0 * eta),
            xi * (2.0 * xi - 1.0),
            eta * (2.0 * eta - 1.0),
            4.0 * xi * (1.0 - xi - eta),
            4.0 * xi * eta,
            4.0 * eta * (1.0 - xi - eta),
        ],
        dtype=float,
    )

    dN_dxi = np.array(
        [
            4.0 * xi + 4.0 * eta - 3.0,
            4.0 * xi - 1.0,
            0.0,
            4.0 * (1.0 - 2.0 * xi - eta),
            4.0 * eta,
            -4.0 * eta,
        ],
        dtype=float,
    )

    dN_deta = np.array(
        [
            4.0 * xi + 4.0 * eta - 3.0,
            0.0,
            4.0 * eta - 1.0,
            -4.0 * xi,
            4.0 * xi,
            4.0 * (1.0 - xi - 2.0 * eta),
        ],
        dtype=float,
    )

    return N, dN_dxi, dN_deta


# ------------------------------------------------------------
# Shape functions: Q9 quadrilateral
# Parent square: -1 <= xi <= 1, -1 <= eta <= 1
# Local order:
#
#       4 ---- 7 ---- 3
#       |             |
#       8      9      6
#       |             |
#       1 ---- 5 ---- 2
#
# ------------------------------------------------------------

def q9_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return N, dN/dxi, dN/deta for the 9-node biquadratic quadrilateral."""
    N = np.array(
        [
            0.25 * xi * eta * (1.0 - xi) * (1.0 - eta),
            -0.25 * xi * eta * (1.0 + xi) * (1.0 - eta),
            0.25 * xi * eta * (1.0 + xi) * (1.0 + eta),
            -0.25 * xi * eta * (1.0 - xi) * (1.0 + eta),
            -0.5 * eta * (1.0 - eta) * (1.0 - xi**2),
            0.5 * xi * (1.0 + xi) * (1.0 - eta**2),
            0.5 * eta * (1.0 + eta) * (1.0 - xi**2),
            -0.5 * xi * (1.0 - xi) * (1.0 - eta**2),
            (1.0 - xi**2) * (1.0 - eta**2),
        ],
        dtype=float,
    )

    dN_dxi = np.array(
        [
            0.25 * eta * (1.0 - eta) * (1.0 - 2.0 * xi),
            -0.25 * eta * (1.0 - eta) * (1.0 + 2.0 * xi),
            0.25 * eta * (1.0 + eta) * (1.0 + 2.0 * xi),
            -0.25 * eta * (1.0 + eta) * (1.0 - 2.0 * xi),
            eta * (1.0 - eta) * xi,
            0.5 * (1.0 + 2.0 * xi) * (1.0 - eta**2),
            -eta * (1.0 + eta) * xi,
            -0.5 * (1.0 - 2.0 * xi) * (1.0 - eta**2),
            -2.0 * xi * (1.0 - eta**2),
        ],
        dtype=float,
    )

    dN_deta = np.array(
        [
            0.25 * xi * (1.0 - xi) * (1.0 - 2.0 * eta),
            -0.25 * xi * (1.0 + xi) * (1.0 - 2.0 * eta),
            0.25 * xi * (1.0 + xi) * (1.0 + 2.0 * eta),
            -0.25 * xi * (1.0 - xi) * (1.0 + 2.0 * eta),
            -0.5 * (1.0 - 2.0 * eta) * (1.0 - xi**2),
            -xi * (1.0 + xi) * eta,
            0.5 * (1.0 + 2.0 * eta) * (1.0 - xi**2),
            xi * (1.0 - xi) * eta,
            -2.0 * eta * (1.0 - xi**2),
        ],
        dtype=float,
    )

    return N, dN_dxi, dN_deta


# ------------------------------------------------------------
# Gauss rules
# ------------------------------------------------------------

def triangle_gauss_points(rule: str = "edge_midpoint") -> list[tuple[float, float, float]]:
    """Return 3-point quadrature points for the parent triangle."""
    if rule == "edge_midpoint":
        return [
            (0.0, 0.5, 1.0 / 6.0),
            (0.5, 0.0, 1.0 / 6.0),
            (0.5, 0.5, 1.0 / 6.0),
        ]

    if rule == "standard":
        return [
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        ]

    raise ValueError(f"Unknown triangle rule: {rule}")


def q9_gauss_points_3x3() -> list[tuple[float, float, float]]:
    """Return 3x3 Gauss quadrature points for the parent square."""
    a = np.sqrt(3.0 / 5.0)
    gp_1d = [-a, 0.0, a]
    w_1d = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

    points = []
    for xi, wx in zip(gp_1d, w_1d):
        for eta, wy in zip(gp_1d, w_1d):
            points.append((xi, eta, wx * wy))
    return points


# ------------------------------------------------------------
# Geometry and connectivity
# ------------------------------------------------------------

def polar(r: float, theta: float) -> tuple[float, float]:
    """Convert polar coordinates to Cartesian coordinates."""
    return r * np.cos(theta), r * np.sin(theta)


def build_nodes_and_connectivity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the one-sector global node table and element connectivity.

    Global nodes are 1-based in the comments/report, but 0-based in Python arrays.

    Node table:
        1  center
        2  r=1,   theta=0
        3  r=1,   theta=45
        4  r=0.5, theta=0
        5  r=1,   theta=22.5
        6  r=0.5, theta=45
        7  r=2,   theta=0
        8  r=2,   theta=45
        9  r=1.5, theta=0
        10 r=2,   theta=22.5
        11 r=1.5, theta=45
        12 r=1.5, theta=22.5
    """

    nodes = np.array(
        [
            (0.0, 0.0),
            polar(1.0, THETA0),
            polar(1.0, THETA1),
            polar(0.5, THETA0),
            polar(1.0, THETAM),
            polar(0.5, THETA1),
            polar(2.0, THETA0),
            polar(2.0, THETA1),
            polar(1.5, THETA0),
            polar(2.0, THETAM),
            polar(1.5, THETA1),
            polar(1.5, THETAM),
        ],
        dtype=float,
    )

    # T6 local nodes: [1,2,3,4,5,6] in the T6 local convention.
    conn_t6 = np.array([0, 1, 2, 3, 4, 5], dtype=int)

    # Q9 local nodes: [1,2,3,4,5,6,7,8,9] in the Q9 local convention.
    conn_q9 = np.array([1, 6, 7, 2, 8, 9, 10, 4, 11], dtype=int)

    return nodes, conn_t6, conn_q9


# ------------------------------------------------------------
# Element integration
# ------------------------------------------------------------

def element_stiffness_force(
    xy: np.ndarray,
    shape_function,
    gauss_points: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """
    Compute scalar Poisson/membrane element matrix and force vector.

    K_ij = integral (N_i,x N_j,x + N_i,y N_j,y) detJ dxi deta
    F_i  = integral N_i detJ dxi deta

    Because TENSION=1 and LOAD_P=1 here, they are kept as explicit factors only.
    """

    nnode = xy.shape[0]
    ke = np.zeros((nnode, nnode), dtype=float)
    fe = np.zeros(nnode, dtype=float)
    diagnostics: list[dict[str, float]] = []

    for xi, eta, w in gauss_points:
        N, dN_dxi, dN_deta = shape_function(xi, eta)

        # Isoparametric derivatives.
        x_xi = np.dot(dN_dxi, xy[:, 0])
        x_eta = np.dot(dN_deta, xy[:, 0])
        y_xi = np.dot(dN_dxi, xy[:, 1])
        y_eta = np.dot(dN_deta, xy[:, 1])

        detJ = x_xi * y_eta - x_eta * y_xi
        if detJ <= 0.0:
            raise ValueError(
                f"Non-positive Jacobian: detJ={detJ:.6e} at xi={xi}, eta={eta}"
            )

        # For J_t = [[x_xi, y_xi],
        #            [x_eta, y_eta]],
        # [N_x, N_y]^T = inv(J_t) [N_xi, N_eta]^T.
        dN_dx = (y_eta * dN_dxi - y_xi * dN_deta) / detJ
        dN_dy = (-x_eta * dN_dxi + x_xi * dN_deta) / detJ

        btb = np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)
        ke += TENSION * btb * detJ * w
        fe += LOAD_P * N * detJ * w

        diagnostics.append(
            {
                "xi": xi,
                "eta": eta,
                "weight": w,
                "detJ": detJ,
                "sumN": float(np.sum(N)),
                "sum_dNdx": float(np.sum(dN_dx)),
                "sum_dNdy": float(np.sum(dN_dy)),
            }
        )

    return ke, fe, diagnostics


# ------------------------------------------------------------
# Assembly and solution
# ------------------------------------------------------------

def assemble_global_system(
    nodes: np.ndarray,
    conn_t6: np.ndarray,
    conn_q9: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Assemble global K and F from the T6 and Q9 elements."""

    ndof = nodes.shape[0]
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    tri_gp = triangle_gauss_points(TRI_RULE)
    q9_gp = q9_gauss_points_3x3()

    ke_t6, fe_t6, diag_t6 = element_stiffness_force(
        nodes[conn_t6, :], t6_shape, tri_gp
    )
    ke_q9, fe_q9, diag_q9 = element_stiffness_force(
        nodes[conn_q9, :], q9_shape, q9_gp
    )

    for conn, ke, fe in [(conn_t6, ke_t6, fe_t6), (conn_q9, ke_q9, fe_q9)]:
        for a_local, A in enumerate(conn):
            F[A] += fe[a_local]
            for b_local, B in enumerate(conn):
                K[A, B] += ke[a_local, b_local]

    diagnostics = {
        "ke_t6": ke_t6,
        "fe_t6": fe_t6,
        "diag_t6": diag_t6,
        "ke_q9": ke_q9,
        "fe_q9": fe_q9,
        "diag_q9": diag_q9,
    }

    return K, F, diagnostics


def solve_system(K: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply u=0 on r=2 boundary nodes and solve the reduced system."""

    # Fixed global nodes are 7, 8, 10 in 1-based numbering -> 6, 7, 9 in Python.
    fixed = np.array([6, 7, 9], dtype=int)
    all_dofs = np.arange(K.shape[0], dtype=int)
    free = np.setdiff1d(all_dofs, fixed)

    U = np.zeros(K.shape[0], dtype=float)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    return U, free, fixed


# ------------------------------------------------------------
# Output utilities
# ------------------------------------------------------------

def write_nodal_csv(nodes: np.ndarray, U: np.ndarray) -> Path:
    """Write nodal solution table to CSV."""
    path = OUT_DIR / "ce526_assignment6_nodal_results.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "x", "y", "r", "theta_deg", "u_fem", "u_exact", "error"])
        for i, ((x, y), u) in enumerate(zip(nodes, U), start=1):
            r = float(np.hypot(x, y))
            theta_deg = float(np.degrees(np.arctan2(y, x))) if r > 1e-14 else 0.0
            ue = float(exact_u(r))
            writer.writerow([i, x, y, r, theta_deg, u, ue, u - ue])

    return path


def print_summary(nodes: np.ndarray, U: np.ndarray, K: np.ndarray, F: np.ndarray) -> None:
    """Print report-style numerical summary."""
    area_exact = 0.5 * R2**2 * (THETA1 - THETA0)

    print("\nNodal solution")
    print("--------------")
    print(f"{'node':>4s} {'x':>12s} {'y':>12s} {'r':>8s} {'u_FEM':>12s} {'u_exact':>12s} {'error':>12s}")
    for i, ((x, y), u) in enumerate(zip(nodes, U), start=1):
        r = float(np.hypot(x, y))
        ue = float(exact_u(r))
        print(f"{i:4d} {x:12.6f} {y:12.6f} {r:8.4f} {u:12.6f} {ue:12.6f} {u-ue:12.6f}")

    print("\nChecks")
    print("------")
    print(f"Triangle rule used             : {TRI_RULE}")
    print(f"sum(F) before BC               : {np.sum(F):.9f}")
    print(f"exact sector area              : {area_exact:.9f}")
    print(f"force-sum error                : {np.sum(F) - area_exact:.9e}")
    print(f"max |row sum of K| before BC   : {np.max(np.abs(np.sum(K, axis=1))):.9e}")
    print(f"symmetry check u2-u3           : {U[1] - U[2]:.9e}")
    print(f"symmetry check u4-u6           : {U[3] - U[5]:.9e}")
    print(f"symmetry check u9-u11          : {U[8] - U[10]:.9e}")


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_radial_2d(nodes: np.ndarray, U: np.ndarray) -> Path:
    """Create a 2D radial plot along theta=0."""

    # theta = 0 line: nodes 1, 4, 2, 9, 7 in 1-based numbering.
    radial_nodes = np.array([0, 3, 1, 8, 6], dtype=int)
    r_fem = np.hypot(nodes[radial_nodes, 0], nodes[radial_nodes, 1])
    u_fem = U[radial_nodes]

    order = np.argsort(r_fem)
    r_fem = r_fem[order]
    u_fem = u_fem[order]

    r_exact = np.linspace(0.0, R2, 300)
    u_exact = exact_u(r_exact)

    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(r_exact, u_exact, label="Exact solution")
    ax.plot(r_fem, u_fem, marker="o", label="FEM nodes on theta = 0")
    ax.set_xlabel("Radius, r")
    ax.set_ylabel("Displacement, u")
    ax.set_title("Radial displacement profile")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    path = OUT_DIR / "ce526_assignment6_radial_2d_plot.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def map_parent_to_physical(
    xi: float,
    eta: float,
    xy: np.ndarray,
    shape_function,
) -> tuple[float, float, np.ndarray]:
    """Map parent coordinates to physical x,y and return N."""
    N, _, _ = shape_function(xi, eta)
    x = float(np.dot(N, xy[:, 0]))
    y = float(np.dot(N, xy[:, 1]))
    return x, y, N


def sample_t6_field(nodes: np.ndarray, conn: np.ndarray, U: np.ndarray, n: int = 24) -> tuple[list[float], list[float], list[float]]:
    """Sample the T6 element field over the parent triangle."""
    xs, ys, us = [], [], []
    xy = nodes[conn]
    ue = U[conn]

    for i in range(n + 1):
        xi = i / n
        for j in range(n + 1 - i):
            eta = j / n
            x, y, N = map_parent_to_physical(xi, eta, xy, t6_shape)
            xs.append(x)
            ys.append(y)
            us.append(float(np.dot(N, ue)))

    return xs, ys, us


def sample_q9_field(nodes: np.ndarray, conn: np.ndarray, U: np.ndarray, n: int = 28) -> tuple[list[float], list[float], list[float]]:
    """Sample the Q9 element field over the parent square."""
    xs, ys, us = [], [], []
    xy = nodes[conn]
    ue = U[conn]

    xis = np.linspace(-1.0, 1.0, n + 1)
    etas = np.linspace(-1.0, 1.0, n + 1)

    for xi in xis:
        for eta in etas:
            x, y, N = map_parent_to_physical(xi, eta, xy, q9_shape)
            xs.append(x)
            ys.append(y)
            us.append(float(np.dot(N, ue)))

    return xs, ys, us


def plot_3d_sector(nodes: np.ndarray, conn_t6: np.ndarray, conn_q9: np.ndarray, U: np.ndarray) -> Path:
    """Create a 3D FEM surface plot over the single 45-degree sector."""

    x1, y1, u1 = sample_t6_field(nodes, conn_t6, U)
    x2, y2, u2 = sample_q9_field(nodes, conn_q9, U)

    x = np.array(x1 + x2, dtype=float)
    y = np.array(y1 + y2, dtype=float)
    u = np.array(u1 + u2, dtype=float)

    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(x, y, u, linewidth=0.2, antialiased=True)
    ax.scatter(nodes[:, 0], nodes[:, 1], U, marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("FEM displacement surface over one 45-degree sector")
    fig.tight_layout()

    path = OUT_DIR / "ce526_assignment6_fem_sector_3d_surface.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    nodes, conn_t6, conn_q9 = build_nodes_and_connectivity()

    K, F, diagnostics = assemble_global_system(nodes, conn_t6, conn_q9)
    U, free, fixed = solve_system(K, F)

    print_summary(nodes, U, K, F)

    nodal_csv = write_nodal_csv(nodes, U)
    radial_png = plot_radial_2d(nodes, U)
    surface_png = plot_3d_sector(nodes, conn_t6, conn_q9, U)

    print("\nFiles written")
    print("-------------")
    print(nodal_csv)
    print(radial_png)
    print(surface_png)


if __name__ == "__main__":
    main()
