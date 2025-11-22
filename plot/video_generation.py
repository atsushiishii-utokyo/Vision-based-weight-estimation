# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


def _ensure_output_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def _draw_point(
    img: np.ndarray,
    coord: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 8,
) -> None:
    """
    Draw a filled circle at coord if it is finite.
    """
    if not np.all(np.isfinite(coord)):
        return
    x, y = int(round(coord[0])), int(round(coord[1]))
    cv2.circle(img, (x, y), radius, color=color, thickness=-1)


def _draw_line(
    img: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """
    Draw a line between p1 and p2 if both are finite.
    """
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        return
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def _draw_box(
    img: np.ndarray,
    box_xyxy: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """
    Draw a rectangle given [x1,y1,x2,y2] if finite.

    Args:
        img:
            Image (BGR).
        box_xyxy:
            (4,) array [x1, y1, x2, y2].
        color:
            BGR color tuple.
        thickness:
            Rectangle thickness.
    """
    if box_xyxy is None or box_xyxy.shape[0] != 4:
        return
    if not np.all(np.isfinite(box_xyxy)):
        return
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def generate_tracking_video(
    video_path: str | Path,
    responses_2d: Dict[str, np.ndarray],
    fps: Optional[float] = None,
    draw_trails: bool = True,
    draw_boxes: bool = True,
) -> Path:
    """
    Generate an MP4 video visualizing tracking results over the original frames.

    Expected keys in responses_2d:
        - "body_front":  (T, 2)
        - "body_rear":   (T, 2)
        - "wheel_front": (T, 2)
        - "wheel_rear":  (T, 2)

    Optional keys for boxes:
        - "vehicle_box":     (T, 4) in [x1,y1,x2,y2]
        - "wheel_front_box": (T, 4) in [x1,y1,x2,y2]
        - "wheel_rear_box":  (T, 4) in [x1,y1,x2,y2]

    Args:
        video_path:
            Path to the original video file.
        responses_2d:
            Tracking result dictionary.
        fps:
            Output FPS. If None, uses the source video's FPS.
        draw_trails:
            Whether to draw motion trails for points.
        draw_boxes:
            Whether to draw bounding boxes for the current frame only.

    Returns:
        Path to the generated MP4 video file.
    """
    output_dir = Path("results/")
    _ensure_output_dir(output_dir)

    video_path = Path(video_path)

    wheel_front = np.asarray(responses_2d["wheel_front"], dtype=float)
    wheel_rear = np.asarray(responses_2d["wheel_rear"], dtype=float)
    body_front = np.asarray(responses_2d["body_front"], dtype=float)
    body_rear = np.asarray(responses_2d["body_rear"], dtype=float)

    n_frames_tracking = wheel_front.shape[0]

    vehicle_box = responses_2d.get("vehicle_box", None)
    wheel_front_box = responses_2d.get("wheel_front_box", None)
    wheel_rear_box = responses_2d.get("wheel_rear_box", None)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0
    out_fps = fps if fps is not None else src_fps

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame from video.")

    height, width = frame.shape[:2]
    size_video = (width, height)

    date_str = datetime.datetime.now().strftime("%m%d")
    output_path = output_dir / f"{date_str}_{video_path.stem}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, out_fps, size_video)

    # Colors (BGR)
    color_wheel_front = (0, 0, 255)
    color_wheel_rear = (0, 255, 0)
    color_body_front = (255, 255, 0)
    color_body_rear = (0, 255, 255)

    color_vehicle_box = (255, 0, 0)
    color_wheel_front_box = (0, 0, 255)
    color_wheel_rear_box = (0, 255, 0)

    frame_idx = 0

    while frame_idx < n_frames_tracking:
        if frame_idx > 0:
            ret, frame = cap.read()
            if not ret:
                break

        # Draw trails (points only)
        if draw_trails and frame_idx > 0:
            for j in range(frame_idx - 1):
                _draw_line(
                    frame, wheel_front[j], wheel_front[j + 1], color_wheel_front, 4
                )
                _draw_line(frame, wheel_rear[j], wheel_rear[j + 1], color_wheel_rear, 4)
                _draw_line(frame, body_front[j], body_front[j + 1], color_body_front, 4)
                _draw_line(frame, body_rear[j], body_rear[j + 1], color_body_rear, 4)

        # Draw current-frame boxes
        if draw_boxes:
            if vehicle_box is not None and frame_idx < len(vehicle_box):
                _draw_box(frame, vehicle_box[frame_idx], color_vehicle_box, 3)
            if wheel_front_box is not None and frame_idx < len(wheel_front_box):
                _draw_box(frame, wheel_front_box[frame_idx], color_wheel_front_box, 3)
            if wheel_rear_box is not None and frame_idx < len(wheel_rear_box):
                _draw_box(frame, wheel_rear_box[frame_idx], color_wheel_rear_box, 3)

        # Draw current points
        _draw_point(frame, wheel_front[frame_idx], color_wheel_front, 8)
        _draw_point(frame, wheel_rear[frame_idx], color_wheel_rear, 8)
        _draw_point(frame, body_front[frame_idx], color_body_front, 8)
        _draw_point(frame, body_rear[frame_idx], color_body_rear, 8)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"Tracking video saved to: {output_path}")
    return output_path


def generate_video_from_frames(
    img_array: list[np.ndarray],
    video_path: str | Path,
    output_dir: str | Path = "results",
    fs: float = 120.0,
    subpixel: bool = True,
    mirror_track: bool = False,
) -> Path:
    """
    Legacy-style helper: create a video directly from a list of frames.

    This is a cleaned-up version of your old snippet that wrote `img_array`
    to disk as a video.

    Args:
        img_array:
            List of frames (H, W, 3) in BGR format.
        video_path:
            Original video path (used only for naming).
        output_dir:
            Directory to save the MP4 file.
        fs:
            Sampling frequency of img_array. We use fs/2 as in your original
            code, so if you want 60 fps, pass fs=120.
        subpixel:
            Add '_nosubpixel' suffix if False.
        mirror_track:
            Add '_mirror_track' suffix if True.

    Returns:
        Path to generated video file.
    """
    output_dir = Path(output_dir)
    _ensure_output_dir(output_dir)
    video_path = Path(video_path)

    if not img_array:
        raise ValueError("img_array is empty; nothing to write.")

    height, width = img_array[0].shape[:2]
    size_video = (width, height)

    dt_now = datetime.datetime.now()
    date_str = dt_now.strftime("%m%d")
    stem = video_path.stem
    name = f"tracked_{date_str}_{stem}"

    if not subpixel:
        name += "_nosubpixel"
    if mirror_track:
        name += "_mirror_track"

    output_path = output_dir / f"{name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, int(fs / 2), size_video)

    for frame in img_array:
        out.write(frame)

    out.release()
    print(f"Video saved to: {output_path}")
    return output_path
