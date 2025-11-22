# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

from pathlib import Path

from estimate_weight import estimate_vehicle_weight
from sim.config import config
from tracker import track_vehicle_motion

video_dir = Path("path/to/video.mp4")

response_dict = track_vehicle_motion(
    video_path=video_dir,
    car_weights="weights/yolov5_car.pt",  # put your trained paths here
    wheel_weights="weights/yolov5_wheel.pt",  # put your trained paths here
    car_img_size=416,
    wheel_img_size=320,
    use_subpixel=True,
)

estimation = estimate_vehicle_weight(
    response=response_dict,
    kf=config.vehicle.kf,
    kr=config.vehicle.kr,
    leng=config.vehicle.length,
    fs=150,
    p=40,
    detrend=False,
    plot_figure=True,
)
