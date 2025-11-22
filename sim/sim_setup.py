# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
from control.matlab import lsim, ss
from numpy import linalg

# =========================
# Dataclasses for parameters
# =========================


@dataclass
class VehicleParams:
    """Vehicle parameters for the half-car model.

    Args:
        mu: mass (t or normalized)
        mf: front tire mass (t or normalized)
        mr: rear tire mass (t or normalized)
        inertia: pitch inertia [kg*m^2]
        lf: distance from CG to front tire [m]
        length: wheelbase [m]
        kf: front suspension stiffness [N/m]
        kr: rear suspension stiffness [N/m]
        ktf: front tire stiffness [N/m]
        ktr: rear tire stiffness [N/m]
        cf: front damping [N*s/m]
        cr: rear damping [N*s/m]
    """

    mu: float  # mass (t or normalized)
    mf: float  # front unsprung mass (t or normalized)
    mr: float  # rear unsprung mass (t or normalized)
    inertia: float  # pitch inertia [kg*m^2]
    lf: float  # distance from CG to front axle [m]
    length: float  # wheelbase [m]

    kf: float
    kr: float
    ktf: float
    ktr: float

    cf: float
    cr: float

    @property
    def lr(self) -> float:
        """Distance from CG to rear axle [m]."""
        return self.length - self.lf


@dataclass
class RoadParams:
    """Road profile / simulation parameters."""

    theta_deg: float
    fh: int  # road profile sampling frequency [Hz]
    length_m: float  # road length [m]
    speed_kmh: float  # vehicle speed [km/h]
    measure_time_s: float
    noise_std_m: float


@dataclass
class SimConfig:
    """Configuration for the simulation.

    Args:
        vehicle: Vehicle parameters
        road: Road parameters
    """

    vehicle: VehicleParams
    road: RoadParams


# ================
# Config utilities
# ================


def load_sim_config(yaml_path: str) -> SimConfig:
    """
    Load simulation configuration (vehicle + road) from a YAML file.

    Args:
        yaml_path: path to the YAML configuration file

    Returns:
        Configuration for the simulation
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    v = cfg["vehicle"]
    r = cfg["road"]

    vehicle = VehicleParams(
        mu=float(v["mu"]),
        mf=float(v["mf"]),
        mr=float(v["mr"]),
        inertia=float(v["inertia"]),
        lf=float(v["lf"]),
        length=float(v["length"]),
        kf=float(v["kf"]),
        kr=float(v["kr"]),
        ktf=float(v["ktf"]),
        ktr=float(v["ktr"]),
        cf=float(v["cf"]),
        cr=float(v["cr"]),
    )

    road = RoadParams(
        theta_deg=float(r["theta_deg"]),
        fh=int(r["fh"]),
        length_m=float(r["length_m"]),
        speed_kmh=float(r["speed_kmh"]),
        measure_time_s=float(r["measure_time_s"]),
        noise_std_m=float(r["noise_std_m"]),
    )

    return SimConfig(vehicle=vehicle, road=road)


# =========================
# Road profile generation
# =========================


def generate_road_profile(
    road: RoadParams,
    vehicle: VehicleParams,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate front and rear road profiles (yf, yr) and time vector t
    given the road and vehicle parameters.

    Args:
        road: Road parameters
        vehicle: Vehicle parameters
        seed: random seed for reproducibility

    Returns:
        Dictionary of road profiles
            "time": time vector (num_samples,)
            "road_profile_front": front road profile (num_samples,)
            "road_profile_rear": rear road profile (num_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Road parameters
    # (1) fh: road profile sampling frequency [Hz]
    # (2) road_len_m: road length [m]
    # (3) speed_mps: vehicle speed [m/s]
    # (4) measure_time: measurement time [s]
    # (5) noise: vertical road profile standard deviation [m]
    fh = road.fh
    road_len_m = road.length_m
    speed_mps = road.speed_kmh * 1000.0 / 3600.0
    measure_time = road.measure_time_s
    noise = road.noise_std_m

    # Base random road profile at 1000 points per meter (mm resolution)
    y = np.random.normal(loc=0.0, scale=noise, size=int(road_len_m * 1000))

    # Distance in mm traveled during one road-profile time step
    r1 = int(speed_mps * 1000.0 / fh)
    if r1 <= 0:
        raise ValueError("Computed r1 <= 0; check speed_kmh and fh in config.")

    # Sample y every r1 indices to get yf
    yf_list = []
    s = 0
    while s < len(y):
        yf_list.append(y[s])
        s += r1
    road_profile_front = np.array(yf_list)

    # Limit yf to measurement time
    num_samples = int(fh * measure_time)
    road_profile_front = road_profile_front[:num_samples]

    # Rear wheel lag (samples)
    lag = vehicle.length / speed_mps * fh
    lag_samples = int(lag)

    # Construct rear profile yr by delaying yf
    road_profile_rear = np.concatenate([np.zeros(lag_samples), road_profile_front])
    road_profile_rear = road_profile_rear[: len(road_profile_front)]

    # Time vector
    t = np.linspace(0.0, measure_time, num_samples)
    assert (
        len(t) == len(road_profile_front) == len(road_profile_rear)
    ), "Time vector and road profiles must have the same length."

    return {
        "time": t,
        "road_profile_front": road_profile_front,
        "road_profile_rear": road_profile_rear,
    }


# =============================
# System matrices & eigen stuff
# =============================


def build_4dof_mck_matrices(
    vehicle: VehicleParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build M, C, K matrices (4DOF) for the half-car model.

    Inputs are the front and rear wheel displacements.
    Outputs are the body motion and the front and rear wheel displacements.

    Args:
        vehicle: vehicle parameters

    Returns:
        M: mass matrix
        C: damping matrix
        K: stiffness matrix
    """
    mu = vehicle.mu
    mf = vehicle.mf
    mr = vehicle.mr
    inertia = vehicle.inertia
    lf = vehicle.lf
    lr = vehicle.lr

    kf = vehicle.kf
    kr = vehicle.kr
    ktf = vehicle.ktf
    ktr = vehicle.ktr

    cf = vehicle.cf
    cr = vehicle.cr

    M = np.array(
        [
            [mu, 0, 0, 0],
            [0, inertia, 0, 0],
            [0, 0, mf, 0],
            [0, 0, 0, mr],
        ]
    )

    C = np.array(
        [
            [cf + cr, lr * cr - lf * cf, -cf, -cr],
            [lr * cr - lf * cf, (lf**2) * cf + (lr**2) * cr, lf * cf, -lr * cr],
            [-cf, lf * cf, cf, 0],
            [-cr, -lr * cr, 0, cr],
        ]
    )

    K = np.array(
        [
            [kf + kr, lr * kr - lf * kf, -kf, -kr],
            [lr * kr - lf * kf, (lf**2) * kf + (lr**2) * kr, lf * kf, -lr * kr],
            [-kf, lf * kf, kf + ktf, 0],
            [-kr, -lr * kr, 0, kr + ktr],
        ]
    )

    return M, C, K


def build_2dof_mck_matrices(
    vehicle: VehicleParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2DOF (body only) M, C, K matrices.
    In this model, we only consider the body motion.
    Inputs are the front and rear wheel displacements.
    Outputs are the body motion.

    Args:
        vehicle: vehicle parameters

    Returns:
        m_2: mass matrix
        c_2: damping matrix
        k_2: stiffness matrix
    """
    mu = vehicle.mu
    inertia = vehicle.inertia
    lf = vehicle.lf
    lr = vehicle.lr

    kf = vehicle.kf
    kr = vehicle.kr
    cf = vehicle.cf
    cr = vehicle.cr

    m_2 = np.array(
        [
            [mu, 0],
            [0, inertia],
        ]
    )

    k_2 = np.array(
        [
            [kf + kr, lr * kr - lf * kf],
            [lr * kr - lf * kf, (lf**2) * kf + (lr**2) * kr],
        ]
    )

    c_2 = np.array(
        [
            [cf + cr, lr * cr - lf * cf],
            [lr * cr - lf * cf, (lf**2) * cf + (lr**2) * cr],
        ]
    )

    return m_2, c_2, k_2


def compute_natural_frequencies(vehicle: VehicleParams) -> Dict[str, np.ndarray]:
    """
    Compute the natural frequencies and damping ratios for the 4DOF and 2DOF models.

    - 4DOF eigenproblem: eig(M^-1 K)
    - 2DOF eigenproblem: eig(M^-1 K)
    - State-space A (4DOF) eigenvalues
    - State-space A2 (2DOF) eigenvalues

    Args:
        vehicle: vehicle parameters

    Returns:
        HCfreq_true_4dof: natural frequencies for the 4DOF model
        HCfreq_true_2dof: natural frequencies for the 2DOF model
        freq_A_4dof: natural frequencies for the 4DOF model
        freq_A_2dof: natural frequencies for the 2DOF model
        zeta: damping ratios for the 4DOF model
        lamda_A: eigenvalues for the 4DOF model
    """
    # 4DOF matrices
    M, C, K = build_4dof_mck_matrices(vehicle)
    Minv = np.linalg.inv(M)

    # 4DOF eigen (undamped)
    lamda2, _ = linalg.eig(np.dot(Minv, K))
    HCfreq_true_4dof = sorted([math.sqrt(lamda) / (2 * math.pi) for lamda in lamda2])

    # 2DOF matrices
    m_2, c_2, k_2 = build_2dof_mck_matrices(vehicle)
    Minv_2 = np.linalg.inv(m_2)

    lamda2_2dof, Phi_2 = linalg.eig(np.dot(Minv_2, k_2))
    HCfreq_true_2dof = sorted(
        [math.sqrt(lamda) / (2 * math.pi) for lamda in lamda2_2dof]
    )

    # State-space A (4DOF)
    A_4, B_4 = build_state_space_matrices(M, C, K)
    lamda_A, _ = linalg.eig(A_4)
    lamda_A_abs = np.abs(lamda_A)
    freq_A_4dof = (
        sorted(lamda_A_abs / (2 * math.pi))[0],
        sorted(lamda_A_abs / (2 * math.pi))[2],
    )

    # Damping ratio for each mode (from A)
    ReD = lamda_A.real
    zeta = -ReD / np.abs(lamda_A)

    # State-space A2 (2DOF)
    A2, B2 = build_state_space_matrices_2dof(m_2, c_2, k_2)
    lamda_A2, Phi_A2 = linalg.eig(A2)
    lamda_A2_abs = np.abs(lamda_A2)
    freq_A_2dof = (
        sorted(lamda_A2_abs / (2 * math.pi))[0],
        sorted(lamda_A2_abs / (2 * math.pi))[2],
    )

    return {
        "HCfreq_true_4dof": HCfreq_true_4dof,
        "HCfreq_true_2dof": HCfreq_true_2dof,
        "freq_A_4dof": freq_A_4dof,
        "freq_A_2dof": freq_A_2dof,
        "zeta": zeta,
        "lamda_A": lamda_A,
    }


# =======================
# State-space & simulation
# =======================


def build_state_space_matrices(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuous-time state-space A, B for the 4DOF system,
    following your original structure.
    """
    Minv = np.linalg.inv(M)
    Z4 = np.zeros((4, 4))
    I4 = np.eye(4)

    A_top = np.concatenate((Z4, I4), axis=1)
    A_bottom = np.concatenate((-np.dot(Minv, K), -np.dot(Minv, C)), axis=1)
    A = np.concatenate((A_top, A_bottom), axis=0)

    B_top = np.zeros((4, 4))
    B_bottom = Minv
    B = np.concatenate((B_top, B_bottom), axis=0)

    return A, B


def build_state_space_matrices_2dof(
    m_2: np.ndarray,
    c_2: np.ndarray,
    k_2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuous-time state-space A, B for the 2DOF (body only) system,
    as in your original A2.

    Inputs are the front and rear wheel displacements.
    Outputs are the body motion.

    Equations:
        x_dot = A2 * x + B2 * u
        y = C2 * x + D2 * u

    Args:
        m_2: mass matrix (2x2)
        c_2: damping matrix (2x2)
        k_2: stiffness matrix (2x2)

    Returns:
        A2: state-space matrix (2x2)
        B2: input-state matrix (2x2)
    """
    Minv_2 = np.linalg.inv(m_2)
    Z2 = np.zeros((2, 2))
    I2 = np.eye(2)

    A2_top = np.concatenate((Z2, I2), axis=1)
    A2_bottom = np.concatenate((-np.dot(Minv_2, k_2), -np.dot(Minv_2, c_2)), axis=1)
    A2 = np.concatenate((A2_top, A2_bottom), axis=0)

    # For completeness, we define B2, though you didn't explicitly use it
    B2_top = np.zeros((2, 2))
    B2_bottom = Minv_2
    B2 = np.concatenate((B2_top, B2_bottom), axis=0)

    return A2, B2


def build_4dof_observation_matrices(
    vehicle: VehicleParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Cc, D for the 4DOF system.

    Observed physical quantities:
    - vertical acceleration at front and rear?
    - pitch angle acceleration, etc.

    Equation:
        y = Cc * x + D * u

    Args:
        vehicle: vehicle parameters

    Returns:
        Cc: observation matrix
        D: input-state matrix
    """
    lf = vehicle.lf
    lr = vehicle.lr

    # 4 states for position, 4 for velocity -> 8 total
    Cc = np.array(
        [
            [1, -lf, 0, 0, 0, 0, 0, 0],
            [1, lr, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    D = np.zeros((4, 4))

    return Cc, D


def create_state_space_system(vehicle: VehicleParams, dof: int = 4) -> ss:
    """
    Build full state-space system (sys) for 4DOF or 2DOF model.

    State-space system:
     - x_dot = A * x + B * u
     - y = Cc * x + D * u

    Args:
        vehicle: vehicle parameters
        dof: number of degrees of freedom
           (Only 4DOF or 2DOF is supported)
    Returns:
        system: state-space system
    """
    # 4DOF matrices
    if dof == 4:
        M, C, K = build_4dof_mck_matrices(vehicle)
        A, B = build_state_space_matrices(M, C, K)
    elif dof == 2:
        m_2, c_2, k_2 = build_2dof_mck_matrices(vehicle)
        A, B = build_state_space_matrices_2dof(m_2, c_2, k_2)
    else:
        raise ValueError(f"Invalid number of degrees of freedom: {dof}")

    Cc, D = build_4dof_observation_matrices(vehicle)

    system = ss(A, B, Cc, D)

    return system


def simulate_response_4dof(
    road_profile: Dict[str, np.ndarray],
    vehicle: VehicleParams,
) -> np.ndarray:
    """
    Simulate the response of the 4DOF half-car model to the road profile.
    Reponses are the vertical displacements of
        - Front body point (index 0)
        - Rear body point (index 1)
        - Front wheel (index 2)
        - Rear wheel (index 3)

    Args:
        road_profile: Dictionary of road profiles
            "time": time vector
            "road_profile_front": front road profile
            "road_profile_rear": rear road profile
        vehicle: vehicle parameters

    Returns:
        response_dict: Dictionary of response
            "time": time vector (N,)
            "body_front": front body point displacement (N,)
            "body_rear": rear body point displacement (N,)
            "wheel_front": front wheel displacement (N,)
            "wheel_rear": rear wheel displacement (N,)
    """
    # Unpack road profile
    t = road_profile["time"]
    road_profile_front = road_profile["road_profile_front"]
    road_profile_rear = road_profile["road_profile_rear"]

    # Create state-space system for 4DOF model
    system: ss = create_state_space_system(vehicle, dof=4)

    # Get tire stiffness to create input matrix u
    ktf = vehicle.ktf
    ktr = vehicle.ktr
    # Input matrix u: shape (4, N)
    # e.g., u = [0, 0, ktf * yf, ktr * yr]
    u = np.zeros((4, len(road_profile_rear)))
    u[2, :] = ktf * np.array(road_profile_front)
    u[3, :] = ktr * np.array(road_profile_rear)

    # lsim expects inputs as (N, nu)
    u_for_lsim = u.T  # (N, 4)
    response, T, xout = lsim(system, U=u_for_lsim, T=t)
    # response (measured output vector): (N, 4)
    # T: (N,) time vector, same as t
    # xout (state vector): (N, 8)
    assert (
        len(response)
        == len(T)
        == len(xout)
        == len(road_profile_front)
        == len(road_profile_rear)
    ), "Time vector and output must have the same length."
    # Make a dictionary of the response like the road_profile
    response_dict = {
        "time": T,
        "body_front": response[:, 0],
        "body_rear": response[:, 1],
        "wheel_front": response[:, 2],
        "wheel_rear": response[:, 3],
    }

    return response_dict
