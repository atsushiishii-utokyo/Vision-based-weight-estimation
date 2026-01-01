# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation
# of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from estimate_weight import estimate_vehicle_weight, extract_vertical_response
from sim.config import config
from tracker import track_vehicle_motion


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace:
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Vision-based vehicle weight estimation from video",
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to input video file",
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=120.0,
        help="Sampling frequency [Hz] (default: 120)",
    )

    parser.add_argument(
        "--p",
        type=int,
        default=40,
        help="SRIM block size p (default: 40)",
    )

    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="Disable subpixel body-point tracking",
    )

    parser.add_argument(
        "--detrend",
        action="store_true",
        help="Apply detrending before SRIM",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot SRIM diagnostics",
    )

    return parser.parse_args()


def run_pipeline(
    video_path: Path,
    fs: float,
    p: int,
    use_subpixel: bool,
    detrend: bool,
    plot_figure: bool,
) -> Dict[str, Any]:
    """
    Run tracking and weight estimation pipeline.

    Args:
        video_path:
            Path to input video.
        fs:
            Sampling frequency [Hz].
        p:
            SRIM block size.
        use_subpixel:
            Whether to use subpixel tracking.
        detrend:
            Whether to detrend signals before SRIM.
        plot_figure:
            Whether to plot SRIM diagnostics.

    Returns:
        dict:
            Estimation results.
    """
    # 1. Track vehicle motion (2D pixel coordinates)
    response_2d = track_vehicle_motion(
        video_path=video_path,
        use_subpixel=use_subpixel,
    )

    # 2. Extract vertical responses
    response_vertical = extract_vertical_response(response_2d)

    # 3. Estimate vehicle weight and related parameters
    estimation = estimate_vehicle_weight(
        response=response_vertical,
        kf=config.vehicle.kf,
        kr=config.vehicle.kr,
        leng=config.vehicle.length,
        fs=fs,
        p=p,
        detrend=detrend,
        plot_figure=plot_figure,
    )

    return estimation


def main() -> None:
    """
    Entry point for command-line execution.
    """
    args = parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    estimation = run_pipeline(
        video_path=video_path,
        fs=args.fs,
        p=args.p,
        use_subpixel=not args.no_subpixel,
        detrend=args.detrend,
        plot_figure=args.plot,
    )

    print("\n===== Estimation Result =====")
    for k, v in estimation.items():
        print(f"{k:>16s}: {v}")


if __name__ == "__main__":
    main()
