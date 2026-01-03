# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy import linalg as la
from sklearn.linear_model import LinearRegression

from plot.plot_results import plot_srim_results
from srim import SrimConfig, SrimResult, srim_realization


@dataclass
class ModeSelectionCriteria:
    """
    Frequency and damping filters for mode selection.

    Args:
        min_frequency_hz:
            Minimum frequency for valid modes [Hz].
        max_frequency_hz:
            Maximum frequency for valid modes [Hz].
        min_damping_ratio:
            Minimum damping ratio for valid modes.
        max_damping_ratio:
            Maximum damping ratio for valid modes.

    Return:
        ModeSelectionCriteria instance.
    """

    min_frequency_hz: float = 0.5
    max_frequency_hz: float = 5.0
    min_damping_ratio: float = 0.0
    max_damping_ratio: float = 0.8

    def build_mask(self, frequency: np.ndarray, damping: np.ndarray) -> np.ndarray:
        """
        Build a boolean mask of modes satisfying the selection criteria.

        Args:
            frequency:
                Modal frequencies [Hz].
            damping:
                Modal damping ratios.

        Return:
            ndarray:
                Boolean mask indicating which modes satisfy the criteria.
        """
        freq_mask = (frequency > self.min_frequency_hz) & (
            frequency < self.max_frequency_hz
        )
        zeta_mask = (damping > self.min_damping_ratio) & (
            damping < self.max_damping_ratio
        )
        return freq_mask & zeta_mask


def estimate_vehicle_weight(
    response: Dict[str, np.ndarray],
    kf: float,
    kr: float,
    leng: float,
    fs: float,
    p: int = 40,
    detrend: bool = False,
    plot_figure: bool = False,
    mode_filter: Optional[ModeSelectionCriteria] = None,
) -> Dict[str, Any]:
    """
    Estimate vehicle weight, pitch inertia, and CG location using SRIM.

    Args:
        response:
            Dictionary containing vibration responses (only vertical displacements)
                - body_front:  front body displacement.
                - body_rear:   rear body displacement.
                - wheel_front: front wheel displacement.
                - wheel_rear:  rear wheel displacement.
            Each is a 1D array sampled at fs.

        kf:
            Front suspension stiffness.

        kr:
            Rear suspension stiffness.

        leng:
            Vehicle wheelbase (lf + lr).

        fs:
            Sampling frequency [Hz].

        p:
            SRIM block size (number of block rows). Default is 40.

        detrend:
            If True, remove linear trend from each input signal.

        plot_figure:
            If True, plot SRIM diagnostics for debugging.

        mode_filter:
            Frequency/damping criteria for mode selection. If None, default
            thresholds are used.

    Return:
        dict:
            A dictionary with:
                - total_weight: estimated sprung mass.
                - moment_inertia: estimated pitch inertia around CG.
                - center_gravity: CG ratio l_G = lf / leng.
                - front_axle: distance lf from front axle to CG [m].
                - rear_axle: distance front axle to rear axle [m].
                - natural_freq: list of two natural frequencies [f1, f2] [Hz].
    """
    if mode_filter is None:
        mode_filter = ModeSelectionCriteria()

    y, z = _prepare_signals_from_response(response=response, detrend=detrend)

    dt = 1.0 / fs
    min_system_order = 4
    row_count = p - min_system_order

    frequency_matrix = np.zeros((row_count, p - 1), dtype="float32")
    damping_matrix = np.zeros((row_count, p - 1), dtype="float32")
    emac_matrix = np.zeros((row_count, p - 1), dtype="float32")
    emac_modes = np.zeros((row_count, 4), dtype="float32")
    mass_inertia_matrix = np.zeros((row_count, 2), dtype="float32")
    cg_ratio_over_order = np.full(row_count, np.nan, dtype="float32")

    srim_config = SrimConfig(block_rows=p)

    # 1. Iterate application of SRIM for each system order and estimate mass/inertia
    for row in range(row_count):
        system_order = row + min_system_order

        srim_result: SrimResult = srim_realization(
            y=y,
            z=z,
            dt=dt,
            system_order=system_order,
            config=srim_config,
        )

        freq_all = np.abs(srim_result.eigenvalues_continuous) / (2.0 * math.pi)
        damping_all = np.asarray(srim_result.damping_ratios)
        emac_all = np.asarray(srim_result.emac_output)
        mode_shapes_y = srim_result.mode_shapes_output

        valid_mask = mode_filter.build_mask(freq_all, damping_all)
        freq_all = freq_all * valid_mask
        emac_all = emac_all * valid_mask

        sort_idx = np.argsort(-emac_all)
        emac_sorted = emac_all[sort_idx]
        freq_sorted = freq_all[sort_idx]
        damping_sorted = damping_all[sort_idx]
        eigen_sorted = srim_result.eigenvalues_continuous[sort_idx]
        mode_shapes_sorted = mode_shapes_y[:, sort_idx]

        frequency_matrix[row, 0:system_order] = freq_sorted
        damping_matrix[row, 0:system_order] = damping_sorted
        emac_matrix[row, 0:system_order] = emac_sorted

        if system_order < 4:
            _invalidate_row(
                row=row,
                frequency_matrix=frequency_matrix,
                emac_matrix=emac_matrix,
                mass_inertia_matrix=mass_inertia_matrix,
                cg_ratio_over_order=cg_ratio_over_order,
            )
            continue

        eigen_four = eigen_sorted[0:4]
        modes_four = mode_shapes_sorted[:, 0:4]
        emac_four = emac_sorted[0:4]

        if np.prod(emac_four) == 0 or np.prod(eigen_four) == 0:
            _invalidate_row(
                row=row,
                frequency_matrix=frequency_matrix,
                emac_matrix=emac_matrix,
                mass_inertia_matrix=mass_inertia_matrix,
                cg_ratio_over_order=cg_ratio_over_order,
            )
            continue

        idx_four = np.argsort(eigen_four)
        eigen_values = eigen_four[idx_four]
        modeshapes = modes_four[:, idx_four]
        emac_four = emac_four[idx_four]

        if not _check_conjugate_pairing(modes_four):
            _invalidate_row(
                row=row,
                frequency_matrix=frequency_matrix,
                emac_matrix=emac_matrix,
                mass_inertia_matrix=mass_inertia_matrix,
                cg_ratio_over_order=cg_ratio_over_order,
            )
            continue

        # Compute A matrix in Cartesian coordinates from eigenvalues and modeshapes
        system_matrix_cartesian = _build_A_matrix_from_modes(
            eigen_values=eigen_values,
            modeshapes=modeshapes,
        )

        # Estimate mass, CG ratio, and inertia from A matrix
        estimation_row = estimate_mass_and_inertia_from_A(
            system_matrix_cartesian=system_matrix_cartesian,
            kf=kf,
            kr=kr,
            leng=leng,
        )

        if not estimation_row["valid"]:
            _invalidate_row(
                row=row,
                frequency_matrix=frequency_matrix,
                emac_matrix=emac_matrix,
                mass_inertia_matrix=mass_inertia_matrix,
                cg_ratio_over_order=cg_ratio_over_order,
            )
            continue

        mass_inertia_matrix[row, 0:2] = np.array(
            [
                estimation_row["estimated_weight"].real,
                estimation_row["estimated_inertia"].real,
            ],
            dtype=float,
        )
        emac_modes[row, 0:4] = emac_four
        cg_ratio_over_order[row] = estimation_row["estimated_center_gravity"].real

    system_orders = np.arange(min_system_order, min_system_order + row_count, 1)

    # 2. Final estimation by aggregating valid results over system orders
    emac_mode_1 = emac_modes[:, 0]
    emac_mode_3 = emac_modes[:, 2]
    emac_avg = (emac_mode_1 + emac_mode_3) / 2.0
    emac_threshold = np.nanmean(emac_avg)
    ## Set a minimum threshold for EMAC
    if emac_threshold < 0.95:
        emac_threshold = 0.95

    valid_idx = np.where(
        (emac_mode_1 > emac_threshold) & ~np.isnan(mass_inertia_matrix[:, 0])
    )[0]

    # If no valid estimates, return empty estimation
    if valid_idx.size == 0:
        estimation = _empty_estimation()
        if plot_figure:
            plot_srim_results(
                system_orders=system_orders,
                frequency_matrix=frequency_matrix,
                damping_matrix=damping_matrix,
                mass_inertia_matrix=mass_inertia_matrix,
                emac_modes=emac_modes,
            )
        return estimation

    # 3. Aggregate valid estimates with sigma clipping
    m_candidates = mass_inertia_matrix[valid_idx, 0]
    i_candidates = mass_inertia_matrix[valid_idx, 1]
    l_g_candidates = cg_ratio_over_order[valid_idx]
    f1_candidates = frequency_matrix[valid_idx, 0]
    f2_candidates = frequency_matrix[valid_idx, 2]

    m_selected = _sigma_clip(m_candidates, sigma1=1.0, sigma2=0.5)
    i_selected = _sigma_clip(i_candidates, sigma1=1.0, sigma2=0.5)

    l_g_selected = float(np.nanmean(l_g_candidates))
    f1_est = float(np.nanmean(f1_candidates))
    f2_est = float(np.nanmean(f2_candidates))

    total_weight = float(np.nanmean(m_selected))
    moment_inertia = float(np.nanmean(i_selected))

    estimation = {
        "total_weight": total_weight,
        "moment_inertia": moment_inertia,
        "center_gravity_ratio": l_g_selected,
        "center_gravity_from_front_axle": l_g_selected * leng,
        "front_axle_weight": (1 - l_g_selected) * total_weight,
        "rear_axle_weight": l_g_selected * total_weight,
        "natural_freq": [f1_est, f2_est],
    }

    if plot_figure:
        plot_srim_results(
            system_orders=system_orders,
            frequency_matrix=frequency_matrix,
            damping_matrix=damping_matrix,
            mass_inertia_matrix=mass_inertia_matrix,
            emac_modes=emac_modes,
        )

    return estimation


def extract_vertical_response(
    response_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Extract only vertical displacement components from the full response dictionary.
    Args:
        response_dict:
            Dictionary containing:
                - body_front:  (N, 2) array of front body displacements [vertical, horizontal].
                - body_rear:   (N, 2) array of rear body displacements [vertical, horizontal].
                - wheel_front: (N, 2) array of front wheel displacements [vertical, horizontal].
                - wheel_rear:  (N, 2) array of rear wheel displacements [vertical, horizontal].
                - time:        (N,) array of time stamps.
    Return:
        dict:
            Dictionary containing only vertical displacements:
                - body_front:  (N,) array of front body vertical displacements.
                - body_rear:   (N,) array of rear body vertical displacements.
                - wheel_front: (N,) array of front wheel vertical displacements.
                - wheel_rear:  (N,) array of rear wheel vertical displacements.
                - time:        (N,) array of time stamps.
    """
    response_dict_vertical = {
        key: value[:, 1] for key, value in response_dict.items() if key != "time"
    }
    response_dict_vertical["time"] = response_dict["time"]

    return response_dict_vertical


def _empty_estimation() -> Dict[str, Any]:
    """
    Create an empty estimation dictionary filled with NaNs.

    Args:
        None

    Return:
        dict:
            Estimation dictionary where all values are NaN.
    """
    nan = float("nan")
    return {
        "total_weight": nan,
        "moment_inertia": nan,
        "center_gravity": nan,
        "front_axle": nan,
        "rear_axle": nan,
        "natural_freq": [nan, nan],
    }


def _prepare_signals_from_response(
    response: Dict[str, np.ndarray],
    detrend: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build body and wheel displacement matrices from the response dictionary.

    Args:
        response:
            Dictionary containing:
                - body_front
                - body_rear
                - wheel_front
                - wheel_rear

        detrend:
            If True, remove linear trend from each signal.

    Return:
        (body_points, wheel_points):
            body_points:
                Array of body displacements [N, 2].
            wheel_points:
                Array of wheel displacements [N, 2].
    """
    body_front = np.asarray(response["body_front"])
    body_rear = np.asarray(response["body_rear"])
    wheel_front = np.asarray(response["wheel_front"])
    wheel_rear = np.asarray(response["wheel_rear"])

    if detrend:
        body_front = _detrending(body_front)
        body_rear = _detrending(body_rear)
        wheel_front = _detrending(wheel_front)
        wheel_rear = _detrending(wheel_rear)

    body_points = np.column_stack((body_front, body_rear))
    wheel_points = np.column_stack((wheel_front, wheel_rear))

    return body_points, wheel_points


def _detrending(data: np.ndarray) -> np.ndarray:
    """
    Remove linear trend from a 1D signal using linear regression.

    Args:
        data:
            Input signal as a 1D array.

    Return:
        ndarray:
            Detrended signal.
    """
    x = np.arange(len(data)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, data)
    trend = model.predict(x)
    return data - trend


def _check_conjugate_pairing(mode_shapes: np.ndarray) -> bool:
    """
    Check whether the first four mode shapes are approximate conjugate pairs.

    Args:
        mode_shapes:
            Modal matrix with at least 4 columns. Modes (0, 1) and (2, 3)
            are assumed to form conjugate pairs.

    Return:
        bool:
            True if both pairs appear conjugate in real part, otherwise False.
    """
    v1 = mode_shapes[:, 0].real
    v1_conj = mode_shapes[:, 1].real
    v2 = mode_shapes[:, 2].real
    v2_conj = mode_shapes[:, 3].real

    cond1 = np.allclose(v1, v1_conj, atol=1e-6)
    cond2 = np.allclose(v2, v2_conj, atol=1e-6)
    return cond1 and cond2


def _build_A_matrix_from_modes(
    eigen_values: np.ndarray,
    modeshapes: np.ndarray,
) -> np.ndarray:
    """
    Build the system matrix A in Cartesian coordinates from eigenvalues and mode shapes.

    Args:
        eigen_values:
            Array of four eigenvalues (complex).
            [λ1, λ2, λ3, λ4]
        modeshapes:
            Corresponding mode shapes (output matrix) for the four modes.
            [v1, v2, v3, v4]
    Return:
        ndarray:
            System matrix A (4x4) in Cartesian coordinates.
    """
    # Build State matrix A in Cartesian coordinates
    # diag_eig = diag([λ1, λ2, λ3, λ4])
    # λ1, λ2: first conjugate pair
    # λ3, λ4: second conjugate pair
    diag_eig = np.diag(eigen_values)
    v = modeshapes
    v_dot = v @ diag_eig
    # Stack mode shapes and their derivatives:
    # phi = [v ; v_dot]
    #     = | v1     v2      v3      v4    |
    #       | λ1*v1  λ2*v2   λ3*v3   λ4*v4 |
    phi = np.vstack((v, v_dot))
    # A = phi ・ diag_eig ・ inv(phi)
    a_cartesian = (phi @ diag_eig) @ la.inv(phi)

    return a_cartesian


def estimate_mass_and_inertia_from_A(
    system_matrix_cartesian: np.ndarray,
    kf: float,
    kr: float,
    leng: float,
) -> Dict[str, Any]:
    """
    Estimate mass, CG ratio, and inertia from A and stiffness.

    Args:
        system_matrix_cartesian:
            System matrix A (4x4) in yf-yr coordinates (continuous-time).
        kf:
            Front suspension stiffness.
        kr:
            Rear suspension stiffness.
        leng:
            Wheelbase (lf + lr).

    Return:
        Dict of esimated values including
            valid:
                False if matrix inversion is ill-conditioned.
            estimated_weight:
                vehicle weight
            esimated_center_gravity:
                CG ratio lf / leng
            estimated_inertia:
                Pitch inertia around CG
    """
    invalid_result = {
        "valid": False,
        "estimated_weight": math.nan,
        "esimated_center_gravity": math.nan,
        "estimated_inertia": math.nan,
    }

    # A_k: bottom-left 2x2 block of A
    # A = |  O    E  |
    #     | A_k  A_c |
    a_k = system_matrix_cartesian[2:4, 0:2]

    cond_number = la.cond(a_k)
    if not np.isfinite(cond_number) or cond_number > 1e12:
        return invalid_result

    # Compute inverse of A_k
    a_k_inv = la.inv(a_k)

    # Estimate mass
    m_u = -kf * (a_k_inv[0, 0] + a_k_inv[0, 1]) - kr * (a_k_inv[1, 0] + a_k_inv[1, 1])
    m_u = float(np.round(m_u.real, 4))

    if abs(m_u) < 1e-8:
        return invalid_result

    # Estimate CG ratio
    l_g = -(a_k_inv[0, 1] * kf + a_k_inv[1, 0] * kr + 2.0 * a_k_inv[1, 1] * kr) / (
        2.0 * m_u
    )
    l_g = float(np.round(l_g.real, 4))

    # Estimate moment of inertia
    i_g = (-a_k_inv[1, 1] * kr - m_u * l_g**2) * leng**2
    i_g = float(np.round(i_g.real, 4))

    return {
        "valid": True,
        "estimated_weight": m_u,
        "estimated_center_gravity": l_g,
        "estimated_inertia": i_g,
    }


def _sigma_clip(values: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    """
    Perform two-stage sigma clipping on an array of values.

    Args:
        values:
            Input array to be sigma-clipped.
        sigma1:
            First-stage threshold in standard deviations.
        sigma2:
            Second-stage threshold in standard deviations.

    Return:
        ndarray:
            Values that passed both clipping stages.
    """
    values = np.asarray(values, dtype=float)
    avg = np.nanmean(values)
    std = np.nanstd(values)
    if std < 1e-3:
        return values

    sigma = (values - avg) / std
    selected = values[sigma < sigma1]

    avg2 = np.nanmean(selected)
    std2 = np.nanstd(selected)
    if std2 < 1e-3:
        return selected

    sigma2_vals = (selected - avg2) / std2
    return selected[sigma2_vals < sigma2]


def _invalidate_row(
    row: int,
    frequency_matrix: np.ndarray,
    emac_matrix: np.ndarray,
    mass_inertia_matrix: np.ndarray,
    cg_ratio_over_order: np.ndarray,
) -> None:
    """
    Mark a row of SRIM results as invalid by filling with NaNs.

    Args:
        row:
            Row index to invalidate.
        frequency_matrix:
            Frequency matrix over system orders.
        emac_matrix:
            EMAC matrix over system orders.
        mass_inertia_matrix:
            Mass and inertia estimates over system orders.
        cg_ratio_over_order:
            CG ratio estimates over system orders.

    Return:
        None
    """
    frequency_matrix[row, :] = np.nan
    emac_matrix[row, :] = np.nan
    mass_inertia_matrix[row, :] = np.nan
    cg_ratio_over_order[row] = np.nan
