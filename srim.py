# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

"""
SRIM (System Realization using Information Matrix) utilities.

Based on:
  J.-N. Juang, "State-Space System Realization With Input- and Output-
  Data Correlation", NASA TP-3622, 1997.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import linalg as la


@dataclass
class SrimConfig:
    """
    Configuration parameters for SRIM realization.

    Args:
        block_rows:
            Number of block rows p of the observability-like matrix Op.
        emac_depth:
            Depth r used for EMAC computation.
        emac_shift:
            Shift index used for EMAC computation.

    Return:
        SrimConfig instance.
    """

    block_rows: int
    emac_depth: int = 5
    emac_shift: int = 1


@dataclass
class SrimResult:
    """
    Result of a SRIM realization for a given system order.

    Args:
        singular_values:
            Singular values of the reduced correlation matrix.
        eigenvalues_discrete:
            Discrete-time eigenvalues of the state matrix A.
        eigenvalues_continuous:
            Continuous-time eigenvalues (log of discrete eigenvalues).
        mode_shapes_output:
            Mode shapes at the physical outputs.
        damping_ratios:
            Modal damping ratios.
        emac_output:
            EMAC values for each mode.

    Return:
        SrimResult instance.
    """

    singular_values: np.ndarray
    eigenvalues_discrete: np.ndarray
    eigenvalues_continuous: np.ndarray
    mode_shapes_output: np.ndarray
    damping_ratios: np.ndarray
    emac_output: np.ndarray


def srim_realization(
    y: np.ndarray,
    z: np.ndarray,
    dt: float,
    system_order: int,
    config: SrimConfig,
) -> SrimResult:
    """
    Perform SRIM-based realization for a given system order.

    Args:
        y:
            Output matrix [y_front, y_rear], shape (N, m).
        z:
            Input matrix [z_front, z_rear], shape (N, r).
        dt:
            Sampling period (Δt).
        system_order:
            Order of the discrete-time state matrix A to retain.
        config:
            SRIM configuration (block rows and EMAC parameters).

    Return:
        SrimResult:
            SRIM realization result containing singular values, eigenvalues
            (discrete and continuous), output mode shapes, damping ratios,
            and EMAC values.
    """
    p = config.block_rows

    yp_block, zp_block, s = _build_block_information_matrices(
        y=y,
        z=z,
        block_rows=p,
    )

    # Correlation matrices (notation close to Juang’s SRIM)
    r_yy = (1.0 / s) * (yp_block @ yp_block.T)
    r_yz = (1.0 / s) * (yp_block @ zp_block.T)
    r_zz = (1.0 / s) * (zp_block @ zp_block.T)

    r_zz_inv = la.inv(r_zz)
    t_yu = r_yz @ r_zz_inv
    r_hh = r_yy - (r_yz @ t_yu.T)

    m_outputs = y.shape[1]
    r_hh_p = r_hh[:, 0 : (p - 1) * m_outputs]

    # SVD of the correlation matrix
    u_left, singular_values, _ = la.svd(r_hh_p, full_matrices=False)

    # Observability-like matrix Op (pm x n): first n columns of U
    op = u_left[:, :system_order]

    # Partition Op into Op1 and Op2 (standard SRIM/ERA structure)
    op1 = op[0 : (p - 1) * m_outputs, :]
    op2 = op[m_outputs : p * m_outputs, :]

    # State matrix A from least-squares solution
    a_discrete = la.pinv(op1) @ op2

    # Output matrix C is first block row of Op
    c_matrix = op[0:m_outputs, :]

    # Eigen-decomposition of A
    eigenvalues_discrete, eigenvectors = la.eig(a_discrete)

    # Modal observability matrix (Op @ V)
    modal_observability = op @ eigenvectors

    # Continuous-time eigenvalues
    eigenvalues_continuous = np.array(
        [cmath.log(ev) / dt for ev in eigenvalues_discrete],
        dtype=complex,
    )

    # Mode shapes at outputs
    mode_shapes_output = c_matrix @ eigenvectors

    # Damping ratios
    real_part = eigenvalues_continuous.real
    damping_ratios = -real_part / np.abs(eigenvalues_continuous)

    emac_output = _compute_emac(
        modal_observability=modal_observability,
        eigenvalues_discrete=eigenvalues_discrete,
        system_order=system_order,
        depth=config.emac_depth,
        shift=config.emac_shift,
        n_outputs=m_outputs,
    )

    return SrimResult(
        singular_values=singular_values,
        eigenvalues_discrete=eigenvalues_discrete,
        eigenvalues_continuous=eigenvalues_continuous,
        mode_shapes_output=mode_shapes_output,
        damping_ratios=damping_ratios,
        emac_output=emac_output,
    )


def _build_block_information_matrices(
    y: np.ndarray,
    z: np.ndarray,
    block_rows: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build block Hankel-like matrices for SRIM information matrix.

    Args:
        y:
            Output matrix, shape (N, m).
        z:
            Input matrix, shape (N, r).
        block_rows:
            Number of block rows p for the stacked blocks.

    Return:
        (Yp, Zp, s):
            Yp:
                Stacked output blocks, shape (m * p, s).
            Zp:
                Stacked input blocks, shape (r * p, s).
            s:
                Number of block columns (N - p).
    """
    n_samples = y.shape[0]
    m_outputs = y.shape[1]
    r_inputs = z.shape[1]

    s = n_samples - block_rows
    if s <= 0:
        raise ValueError(
            f"Not enough samples for block_rows={block_rows}: "
            f"need at least p+1 samples."
        )

    yp_block = np.zeros((m_outputs * block_rows, s), dtype=float)
    zp_block = np.zeros((r_inputs * block_rows, s), dtype=float)

    for i in range(block_rows):
        y_slice = y[i : i + s, :].T
        z_slice = z[i : i + s, :].T
        yp_block[m_outputs * i : m_outputs * (i + 1), :] = y_slice
        zp_block[r_inputs * i : r_inputs * (i + 1), :] = z_slice

    return yp_block, zp_block, s


def _compute_emac(
    modal_observability: np.ndarray,
    eigenvalues_discrete: np.ndarray,
    system_order: int,
    depth: int,
    shift: int,
    n_outputs: int,
) -> np.ndarray:
    """
    Compute Extended Modal Amplitude Coherence (EMAC) for each mode.

    Args:
        modal_observability:
            Modal observability matrix Op V, shape (p * m, n).
        eigenvalues_discrete:
            Discrete-time eigenvalues of A.
        system_order:
            System order n.
        depth:
            EMAC depth parameter r.
        shift:
            EMAC shift index.
        n_outputs:
            Number of physical outputs m.

    Return:
        ndarray:
            EMAC values as percentages for each mode, shape (n,).
    """
    initial_block = modal_observability[0:n_outputs, :]
    shifted_observed = np.zeros((n_outputs, system_order), dtype=complex)

    for j in range(system_order):
        shifted_observed[:, j] = initial_block[:, j] * (
            eigenvalues_discrete[j] ** (depth + shift)
        )

    row_start = (depth + shift) * n_outputs
    row_end = (depth + shift + 1) * n_outputs
    shift_block = modal_observability[row_start:row_end, :]

    # Amplitude ratio
    amplitude_ratio = np.abs(shift_block) / np.abs(shifted_observed)

    over_one_mask = amplitude_ratio > 1.0
    amplitude_ratio[over_one_mask] = 1.0 / amplitude_ratio[over_one_mask]

    # Phase weighting
    phase_shift = np.zeros_like(shift_block, dtype=complex)
    rows, cols = shift_block.shape
    for i in range(rows):
        for j in range(cols):
            phase_shift[i, j] = cmath.phase(shift_block[i, j] / shifted_observed[i, j])

    phase_weight = np.zeros((n_outputs, system_order), dtype=float)
    for j in range(system_order):
        for k in range(n_outputs):
            angle = abs(phase_shift[k, j])
            if angle <= (math.pi / 4.0):
                phase_weight[k, j] = 1.0 - angle / (math.pi / 4.0)
            else:
                phase_weight[k, j] = 0.0

    emac_shift = amplitude_ratio * phase_weight
    emac_shift = emac_shift.T

    weight = np.abs(initial_block) ** 2
    emac_values = np.diag(emac_shift @ weight)

    sum_weight = np.sum(weight, axis=0)
    emac_values = 100.0 * (emac_values / sum_weight.T)

    return emac_values
