# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def resample_simulation(
    road_profile: Dict[str, np.ndarray],
    response_dict: Dict[str, np.ndarray],
    fs: float = 300.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Resample road profile and responses onto a common uniform time grid
    at sampling frequency fs. No cropping by Start/Measurement_Time is applied.

    Args:
        road_profile:
            Dict with:
                "time"
                "road_profile_front"
                "road_profile_rear"
        response_dict:
            Dict with:
                "time"
                "body_front"
                "body_rear"
                "wheel_front"
                "wheel_rear"
        fs:
            Target sampling frequency [Hz].

    Returns:
        resampled_road_profiles, resampled_responses
            resampled_road_profiles:
                "time"
                "road_profile_front"
                "road_profile_rear"
            resampled_responses:
                "time"
                "body_front"
                "body_rear"
                "wheel_front"
                "wheel_rear"
    """
    # Original time vectors
    t_rp = np.asarray(road_profile["time"])
    t_resp = np.asarray(response_dict["time"])

    # Determine overlapping time range
    t_min = max(t_rp[0], t_resp[0])
    t_max = min(t_rp[-1], t_resp[-1])

    if t_max <= t_min:
        raise ValueError(
            "Road profile and response do not have an overlapping time range."
        )

    dt = 1.0 / fs
    t_new = np.arange(t_min, t_max, dt)

    # Resample road profile
    rp_front = np.interp(t_new, t_rp, road_profile["road_profile_front"])
    rp_rear = np.interp(t_new, t_rp, road_profile["road_profile_rear"])

    # Resample responses
    body_front = np.interp(t_new, t_resp, response_dict["body_front"])
    body_rear = np.interp(t_new, t_resp, response_dict["body_rear"])
    wheel_front = np.interp(t_new, t_resp, response_dict["wheel_front"])
    wheel_rear = np.interp(t_new, t_resp, response_dict["wheel_rear"])

    resampled_road_profiles = {
        "time": t_new,
        "road_profile_front": rp_front,
        "road_profile_rear": rp_rear,
    }

    resampled_responses = {
        "time": t_new,
        "body_front": body_front,
        "body_rear": body_rear,
        "wheel_front": wheel_front,
        "wheel_rear": wheel_rear,
    }

    return resampled_road_profiles, resampled_responses


def plot_simulation_results(
    road_profile: Dict[str, np.ndarray],
    responses: Dict[str, np.ndarray],
    Start: float = 0.0,
    Measurement_Time: float = 10.0,
) -> None:
    """
    Crop rresponses signals to [Start, Start + Measurement_Time)
    and plot road profile, wheel displacement, and body displacement.

    Args:
        road_profile:
            Dict with:
                "time", "road_profile_front", "road_profile_rear"
        responses:
            Output of responses, with keys:
                "time",
                "body_front","body_rear",
                "wheel_front","wheel_rear"
        Start:
            Plot start time [s]
        Measurement_Time:
            Duration to plot [s]
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    t = road_profile["time"]
    t_start = Start
    t_end = Start + Measurement_Time

    # Sanity check: interval must be inside resampled time range
    assert (
        t[0] <= t_start and t[-1] >= t_end
    ), "Resampled data do not cover [Start, Start + Measurement_Time]."

    # Time mask for cropping
    mask = (t >= t_start) & (t < t_end)

    # Time axis shifted to start at 0
    t_plot = t[mask] - Start

    # Crop signals
    zf = road_profile["road_profile_front"][mask]
    zr = road_profile["road_profile_rear"][mask]
    wf = responses["wheel_front"][mask]
    wr = responses["wheel_rear"][mask]
    bf = responses["body_front"][mask]
    br = responses["body_rear"][mask]

    # ========== Plotting ===========
    _, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    # (a) Road profile
    ax0.plot(t_plot, zf, color="blue", label="front")
    ax0.plot(t_plot, zr, color="red", linestyle="--", label="rear")
    ax0.set_title("(a) Road profile", fontsize=25)
    ax0.legend(fontsize=15)
    ax0.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax0.set_yticklabels([-0.2, -0.1, 0, 0.1, 0.2], fontsize=15)

    # (b) Wheel displacement
    ax1.plot(t_plot, wf, color="blue", label="front")
    ax1.plot(t_plot, wr, color="red", linestyle="--", label="rear")
    ax1.set_title("(b) Wheel displacement", fontsize=25)
    ax1.legend(fontsize=15)
    ax1.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax1.set_yticklabels([-0.2, -0.1, 0, 0.1, 0.2], fontsize=15)

    # (c) Body point displacement
    ax2.plot(t_plot, bf, color="blue", label="front")
    ax2.plot(t_plot, br, color="red", linestyle="--", label="rear")
    ax2.set_title("(c) Body point displacement", fontsize=25)
    ax2.set_xlabel("Time (s)", fontsize=20)
    ax2.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax2.set_yticklabels([-0.2, -0.1, 0, 0.1, 0.2], fontsize=15)
    ax2.legend(fontsize=15)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels([0, 1, 2, 3], fontsize=15)

    plt.tight_layout()
    plt.show()


def plot_srim_results(
    system_orders: np.ndarray,
    frequency_matrix: np.ndarray,
    damping_matrix: np.ndarray,
    mass_inertia_matrix: np.ndarray,
    emac_modes: np.ndarray,
) -> None:
    """
    Plot SRIM diagnostics with consistent color style for conjugate mode pairs.

    Args:
        system_orders:
            Array of system orders used during SRIM (formerly Ncondense).
            Shape: [n_rows].

        frequency_matrix:
            Matrix of estimated natural frequency.
            Shape: [n_rows, n_modes].

        damping_matrix:
            Matrix of estimated damping ratios.
            Shape: [n_rows, n_modes].

        mass_inertia_matrix:
            Mass/inertia matrix. Columns:
                - 0: mass estimate
                - 1: inertia estimate
                - 2-3: off-diagonal terms |M_xy|, |M_yx|

        emac_modes:
            EMAC values for modes 1-4.
            Shape: [n_rows, 4].

    Return:
        None
    """

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    _, (ax_freq, ax_zeta, ax_mass_inertia, ax_emac) = plt.subplots(
        4, 1, sharex=True, figsize=(10, 10)
    )
    plt.subplots_adjust(hspace=0.4)

    # ----------------------------------------------------------
    # Color groups for conjugate pairs:
    #   Pair 1: modes (1,2) → blue
    #   Pair 2: modes (3,4) → red
    # ----------------------------------------------------------
    pair_colors = ["blue", "red"]
    n_modes = min(4, frequency_matrix.shape[1])

    # ----------------------------------------------------------
    # (a) Natural frequency
    # ----------------------------------------------------------
    print_once = [False, False]

    for mode in range(n_modes):
        pair_idx = mode // 2
        color = pair_colors[pair_idx]

        freq_vals = frequency_matrix[:, mode].real
        valid = ~np.isnan(freq_vals)
        if not np.any(valid):
            continue

        label = None
        if not print_once[pair_idx]:
            label = f"Mode pair {pair_idx + 1}"
            print_once[pair_idx] = True

        ax_freq.scatter(
            system_orders[valid],
            freq_vals[valid],
            marker="p",
            color=color,
            label=label,
        )

    ax_freq.set_title("(a) Natural frequency", fontsize=25)
    ax_freq.set_ylabel("Frequency (Hz)", fontsize=20)
    ax_freq.set_ylim([0, 3])
    ax_freq.set_yticks(np.arange(0.0, 3.5, 0.5))
    ax_freq.grid(True)
    ax_freq.legend(fontsize=12)
    ax_freq.tick_params(labelsize=15)

    # ----------------------------------------------------------
    # (b) Damping ratio
    # ----------------------------------------------------------
    print_once = [False, False]

    for mode in range(min(4, damping_matrix.shape[1])):
        pair_idx = mode // 2
        color = pair_colors[pair_idx]

        zeta_vals = damping_matrix[:, mode].real
        valid = ~np.isnan(zeta_vals)

        if not np.any(valid):
            continue

        label = None
        if not print_once[pair_idx]:
            label = f"Mode pair {pair_idx + 1}"
            print_once[pair_idx] = True

        ax_zeta.scatter(
            system_orders[valid],
            zeta_vals[valid],
            marker="p",
            color=color,
            label=label,
        )

    ax_zeta.set_title("(b) Damping ratio", fontsize=25)
    ax_zeta.set_ylabel("ζ", fontsize=20)
    ax_zeta.set_ylim([0, 1])
    ax_zeta.grid(True)
    ax_zeta.legend(fontsize=12)
    ax_zeta.tick_params(labelsize=15)

    # ----------------------------------------------------------
    # (c) Mass / Inertia
    # ----------------------------------------------------------
    mass_vals = mass_inertia_matrix[:, 0].real
    inertia_vals = mass_inertia_matrix[:, 1].real

    if np.any(~np.isnan(mass_vals)):
        ax_mass_inertia.scatter(
            system_orders[~np.isnan(mass_vals)],
            mass_vals[~np.isnan(mass_vals)],
            marker="p",
            color="blue",
            label="Mass",
        )

    if np.any(~np.isnan(inertia_vals)):
        ax_mass_inertia.scatter(
            system_orders[~np.isnan(inertia_vals)],
            inertia_vals[~np.isnan(inertia_vals)],
            marker="p",
            color="red",
            label="Inertia",
        )

    ax_mass_inertia.set_title("(c) Estimated mass / inertia", fontsize=25)
    ax_mass_inertia.set_ylabel("Value", fontsize=20)
    ax_mass_inertia.set_ylim([0, 3.0])
    ax_mass_inertia.grid(True)
    ax_mass_inertia.legend(fontsize=15)
    ax_mass_inertia.tick_params(labelsize=15)
    ax_mass_inertia.set_yticks(np.arange(0.0, 3.5, 0.5))

    # ----------------------------------------------------------
    # (d) EMAC — representative modes for the conjugate pairs
    #     Pair 1  → emac_modes[:, 0]
    #     Pair 2  → emac_modes[:, 2]
    # ----------------------------------------------------------
    width = 0.35

    color_pair1 = "#4A90E2"  # dark blue
    color_pair2 = "#D0021B"  # dark red

    emac_pair1 = np.nan_to_num(emac_modes[:, 0].real)
    emac_pair2 = np.nan_to_num(emac_modes[:, 2].real)

    ax_emac.bar(
        system_orders - width / 2,
        emac_pair1,
        width,
        color=color_pair1,
        label="Pair 1 (vertical)",
    )
    ax_emac.bar(
        system_orders + width / 2,
        emac_pair2,
        width,
        color=color_pair2,
        label="Pair 2 (pitch)",
    )

    ax_emac.set_title("(d) EMAC values", fontsize=25)
    ax_emac.set_xlabel("System order", fontsize=20)
    ax_emac.set_ylabel("EMAC (%)", fontsize=20)
    ax_emac.set_ylim([0, 100])
    ax_emac.grid(True)
    ax_emac.legend(fontsize=15)
    ax_emac.tick_params(labelsize=15)


def plot_tracking_results(
    responses_2d: Dict[str, np.ndarray],
    fs: float = 120.0,
    Start: float = 0.0,
    Measurement_Time: float = 2.0,
    flip_y: bool = True,
    normalize: bool = True,
) -> None:
    """
    Plot tracking results (from video) for body and wheel vertical motion
    based on the response_dict returned by the tracking pipeline.

    The function extracts the vertical (y) component from 2D coordinates
    and plots:
        (a) Wheel vertical displacement (front/rear)
        (b) Body vertical displacement (front/rear)

    Args:
        responses_2d:
            Dict with keys:
                "body_front", "body_rear",
                "wheel_front", "wheel_rear"
            Each value is an array of shape (T, 2), where [:, 0] is x [px],
            [:, 1] is y [px] in image coordinates (y downward).
            Optionally may contain "time": array of shape (T,).
        fs:
            Sampling frequency [Hz] used to build time axis if "time"
            is not present in responses_2d.
        Start:
            Plot start time [s].
        Measurement_Time:
            Duration to plot [s]. If None, plots to the end.
        flip_y:
            If True, invert sign of y so that upward motion becomes positive.
        normalize:
            If True, subtract the initial mean value (first 0.5 s or up to
            50 samples) to show displacement relative to the initial position.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # ---- 1. Build time axis ----------------------------------------------
    if "time" in responses_2d:
        t = np.asarray(responses_2d["time"], dtype=float)
        dt_est = np.mean(np.diff(t))
        if fs is None:
            fs = 1.0 / dt_est
    else:
        # infer length from any signal
        key0 = "body_front"
        T = responses_2d[key0].shape[0]
        t = np.arange(T, dtype=float) / fs

    if Measurement_Time is None:
        Measurement_Time = float(t[-1] - t[0])

    t_start = Start
    t_end = Start + Measurement_Time

    if t_start < t[0] or t_end > t[-1]:
        t_start = max(t_start, t[0])
        t_end = min(t_end, t[-1])

    mask = (t >= t_start) & (t < t_end)
    if not np.any(mask):
        raise ValueError("No samples in the requested time window.")

    t_plot = t[mask] - t_start

    # ---- 2. Helper to process vertical signals ---------------------------
    def _prepare_vertical(signal_2d: np.ndarray) -> np.ndarray:
        """
        Extract and preprocess vertical (y) component from a (T, 2) array.
        """
        if signal_2d.ndim != 2 or signal_2d.shape[1] < 2:
            raise ValueError(
                f"Expected shape (T, 2) for tracking data, got {signal_2d.shape}."
            )

        y = np.asarray(signal_2d[:, 1], dtype=float)

        # Normalize by initial mean (first 0.5 s or 50 samples)
        if normalize:
            n0 = min(50, len(y))
            # If we have time array, use min(0.5 s, full)
            if "time" in responses_2d:
                t0 = t - t[0]
                n0 = np.count_nonzero(t0 <= 0.5)
                if n0 == 0:
                    n0 = min(50, len(y))
            baseline = np.nanmean(y[:n0])
            y = y - baseline

        if flip_y:
            y = -y

        return y[mask]

    # ---- 3. Extract vertical signals -------------------------------------
    wf = _prepare_vertical(responses_2d["wheel_front"])
    wr = _prepare_vertical(responses_2d["wheel_rear"])
    bf = _prepare_vertical(responses_2d["body_front"])
    br = _prepare_vertical(responses_2d["body_rear"])

    # ---- 4. Plot ---------------------------------------------------------
    fig, (ax_wheel, ax_body) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # (a) Wheel vertical displacement
    ax_wheel.plot(t_plot, wf, color="blue", label="front wheel")
    ax_wheel.plot(t_plot, wr, color="red", linestyle="--", label="rear wheel")
    ax_wheel.set_title("(a) Tracked wheel vertical displacement", fontsize=20)
    if normalize:
        ax_wheel.set_ylabel("Displacement [px or rel.]", fontsize=16)
    else:
        ax_wheel.set_ylabel("Vertical position [px]", fontsize=16)
    ax_wheel.legend(fontsize=12)
    ax_wheel.tick_params(labelsize=12)
    ax_wheel.grid(True)

    # (b) Body vertical displacement
    ax_body.plot(t_plot, bf, color="blue", label="front body point")
    ax_body.plot(t_plot, br, color="red", linestyle="--", label="rear body point")
    ax_body.set_title("(b) Tracked body vertical displacement", fontsize=20)
    ax_body.set_xlabel("Time (s)", fontsize=16)
    if normalize:
        ax_body.set_ylabel("Displacement [px or rel.]", fontsize=16)
    else:
        ax_body.set_ylabel("Vertical position [px]", fontsize=16)
    ax_body.legend(fontsize=12)
    ax_body.tick_params(labelsize=12)
    ax_body.grid(True)

    plt.tight_layout()
    plt.show()
