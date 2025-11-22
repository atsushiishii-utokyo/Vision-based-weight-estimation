# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from detectors import DetectionResult, Detector_car, Detector_wheels
from track_utils import (
    BackgroundShift,
    error_warning,
    estimate_background_shift,
    find_body_point_and_wheel,
    match_template_box,
)

IntPoint = Tuple[int, int]
Point = Tuple[float, float]
Box = Tuple[IntPoint, IntPoint]


@dataclass
class FrameDetections:
    """
    Container for per-frame detections and background shift.

    Attributes:
        vehicle_box:
            Vehicle bounding box in full-image coordinates.
        wheel_front_box:
            Front wheel bounding box in full-image coordinates (may be None).
        wheel_rear_box:
            Rear wheel bounding box in full-image coordinates (may be None).
        shift:
            Estimated background shift between frames.
    """

    vehicle_box: Box | None
    wheel_front_box: Box | None
    wheel_rear_box: Box | None
    shift: BackgroundShift


def _majority_label(labels: List[str]) -> str:
    """
    Compute the most frequent label in a list.

    Args:
        labels:
            List of string labels.

    Returns:
        Most frequent label, or "car" if empty.
    """
    if not labels:
        return "car"
    uniq, counts = np.unique(labels, return_counts=True)
    return str(uniq[int(np.argmax(counts))])


def _box_area(b: Box) -> float:
    """
    Compute the area of a bounding box.

    Args:
        b:
            Bounding box ((x1,y1),(x2,y2)).

    Returns:
        Box area in pixels^2.
    """
    (x1, y1), (x2, y2) = b
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _box_center(b: Box) -> Tuple[float, float]:
    """
    Compute the center of a bounding box.

    Args:
        b:
            Bounding box ((x1,y1),(x2,y2)).

    Returns:
        (cx, cy) center coordinates.
    """
    (x1, y1), (x2, y2) = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _box_iou(a: Box, b: Box) -> float:
    """
    Compute intersection-over-union (IoU) between two boxes.

    Args:
        a:
            First box.
        b:
            Second box.

    Returns:
        IoU in [0, 1].
    """
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    ua = _box_area(a) + _box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def _select_consistent_vehicle(
    locs: List[Box],
    scores: List[float],
    labels: List[str] | None,
    prev_box: Box | None,
    frame_shape: Tuple[int, int, int],
) -> Tuple[Box, str]:
    """
    Select a vehicle box while maintaining temporal consistency.

    If prev_box is None, the max-score detection is selected.
    Otherwise, the selection favors high IoU with prev_box, small center distance,
    and similar box area, with detection score as a secondary factor.

    Args:
        locs:
            Candidate bounding boxes.
        scores:
            Detection scores for each candidate.
        labels:
            Optional detection labels aligned with locs/scores.
        prev_box:
            Vehicle box from previous frame, if available.
        frame_shape:
            Frame shape (H, W, C).

    Returns:
        (selected_box, selected_label)
    """
    if len(locs) == 0:
        raise ValueError("No candidate boxes")

    if prev_box is None:
        i = int(np.argmax(np.asarray(scores, dtype=float)))
        lab = labels[i] if labels else "car"
        return locs[i], (lab or "car")

    H, W = frame_shape[:2]
    prev_cx, prev_cy = _box_center(prev_box)
    prev_area = _box_area(prev_box)

    best_i = 0
    best_val = -1e18

    w_iou = 3.0
    w_dist = 2.0
    w_area = 1.0
    w_score = 0.5

    max_diag = float(np.hypot(W, H) + 1e-6)

    for i, b in enumerate(locs):
        iou = _box_iou(prev_box, b)

        cx, cy = _box_center(b)
        dist = float(np.hypot(cx - prev_cx, cy - prev_cy)) / max_diag

        area = _box_area(b)
        if prev_area > 1e-6 and area > 1e-6:
            area_term = abs(np.log(area / prev_area))
        else:
            area_term = 10.0

        sc = float(scores[i])

        val = (w_iou * iou) - (w_dist * dist) - (w_area * area_term) + (w_score * sc)

        if iou >= 0.2:
            val += 1.0

        if val > best_val:
            best_val = val
            best_i = i

    lab = labels[best_i] if labels else "car"
    return locs[best_i], (lab or "car")


def _extract_wheel_boxes(
    img_car: np.ndarray,
    car_box: Box,
    wheel_result: DetectionResult,
) -> Tuple[Box | None, Box | None]:
    """
    Convert wheel detections inside cropped car image to full-image boxes.

    Args:
        img_car:
            Cropped car image (used for width reference in 1-wheel case).
        car_box:
            Car bounding box in full-image coordinates.
        wheel_result:
            Wheel detection result in cropped-car coordinates.

    Returns:
        (front_wheel_box, rear_wheel_box) in full-image coordinates.
    """
    loc_wheels = wheel_result.locs
    if loc_wheels is None or len(loc_wheels) == 0:
        return None, None

    (car_x1, car_y1), _ = car_box

    if len(loc_wheels) > 2:
        areas = []
        for tl, br in loc_wheels:
            w = br[0] - tl[0]
            h = br[1] - tl[1]
            areas.append(w * h)
        idx = np.argsort(areas)[::-1]
        loc_wheels = [loc_wheels[i] for i in idx[:2]]

    loc_wheels.sort(key=lambda b: b[0][0])

    if len(loc_wheels) == 1:
        (tl, br) = loc_wheels[0]
        cx = (tl[0] + br[0]) / 2.0
        car_w = img_car.shape[1]
        box = ((car_x1 + tl[0], car_y1 + tl[1]), (car_x1 + br[0], car_y1 + br[1]))
        if cx < car_w / 2.0:
            return box, None
        return None, box

    (tl_f, br_f), (tl_r, br_r) = loc_wheels[:2]
    front_box = (
        (car_x1 + tl_f[0], car_y1 + tl_f[1]),
        (car_x1 + br_f[0], car_y1 + br_f[1]),
    )
    rear_box = (
        (car_x1 + tl_r[0], car_y1 + tl_r[1]),
        (car_x1 + br_r[0], car_y1 + br_r[1]),
    )
    return front_box, rear_box


def run_detection(
    video_path: str | Path,
    car_detector: Detector_car,
    wheel_detector: Detector_wheels,
) -> tuple[List[FrameDetections], str]:
    """
    Pass 1: Run vehicle detection for all frames and wheel detection inside the chosen vehicle box.

    Args:
        video_path:
            Path to the input video.
        car_detector:
            Vehicle detector.
        wheel_detector:
            Wheel detector (expects cropped vehicle image).

    Returns:
        (frame_detections, majority_label)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: List[FrameDetections] = []
    labels_hist: List[str] = []

    prev_frame: np.ndarray | None = None
    prev_car_patch: np.ndarray | None = None
    prev_vehicle_box: Box | None = None

    for _ in tqdm(
        range(n_frames),
        desc="Pass 1/2: Detect vehicle+wheels & background shift",
        unit="frame",
    ):
        ret, frame = cap.read()
        if not ret:
            break

        if error_warning(frame):
            break

        if prev_frame is None:
            shift = BackgroundShift(dx=0.0, dy=0.0)
        else:
            shift = estimate_background_shift(prev_frame, frame)

        car_result = car_detector.detect_car(frame)

        vehicle_box: Box | None = None
        label = "car"

        if (
            car_result.locs is not None
            and car_result.scores is not None
            and len(car_result.locs) > 0
        ):
            vehicle_box, label = _select_consistent_vehicle(
                locs=car_result.locs,
                scores=car_result.scores,
                labels=car_result.labels if hasattr(car_result, "labels") else None,
                prev_box=prev_vehicle_box,
                frame_shape=frame.shape,
            )
            labels_hist.append(label)
        else:
            if prev_car_patch is not None and prev_car_patch.size > 0:
                tl, br = match_template_box(frame, prev_car_patch)
                vehicle_box = (tl, br)

        wheel_front: Box | None = None
        wheel_rear: Box | None = None

        if vehicle_box is not None:
            (x1, y1), (x2, y2) = vehicle_box
            img_car = frame[y1:y2, x1:x2]
            if img_car.size > 0:
                prev_car_patch = img_car.copy()
                prev_vehicle_box = vehicle_box
                wheel_result = wheel_detector.detect_wheels(img_car)
                wheel_front, wheel_rear = _extract_wheel_boxes(
                    img_car, vehicle_box, wheel_result
                )

        frames.append(
            FrameDetections(
                vehicle_box=vehicle_box,
                wheel_front_box=wheel_front,
                wheel_rear_box=wheel_rear,
                shift=shift,
            )
        )
        prev_frame = frame

    cap.release()
    majority_label = _majority_label(labels_hist)
    print(f"Majority vehicle label: {majority_label}")
    return frames, majority_label


def _box_to_xyxy_or_nan(box: Box | None) -> np.ndarray:
    """
    Convert a Box to [x1,y1,x2,y2] or NaNs.

    Args:
        box:
            Box in ((x1,y1),(x2,y2)) format or None.

    Returns:
        (4,) float array.
    """
    if box is None:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    (x1, y1), (x2, y2) = box
    return np.array([float(x1), float(y1), float(x2), float(y2)], dtype=np.float32)


def track_vehicle_motion(
    video_path: str | Path,
    use_subpixel: bool = True,
) -> dict[str, np.ndarray]:
    """
    Track vehicle body and wheel motions from a video.

    Args:
        video_path:
            Path to input video file.
        use_subpixel:
            Whether to use subpixel refinement for wheel centers.

    Returns:
        Dictionary with keys:
            'body_front': (N, 2) float array of stabilized body front points.
            'body_rear': (N, 2) float array of stabilized body rear points.
            'wheel_front': (N, 2) float array of stabilized front wheel centers.
            'wheel_rear': (N, 2) float array of stabilized rear wheel centers.
            'vehicle_box': (N, 4) float array of [x1,y1,x2,y2] (full-image).
            'wheel_front_box': (N, 4) float array of [x1,y1,x2,y2] (full-image).
            'wheel_rear_box': (N, 4) float array of [x1,y1,x2,y2] (full-image).
    """
    video_path = Path(video_path)

    car_detector = Detector_car()
    wheel_detector = Detector_wheels()

    frame_dets, majority_label = run_detection(video_path, car_detector, wheel_detector)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = len(frame_dets)

    body_front_list: List[List[float]] = []
    body_rear_list: List[List[float]] = []
    wheel_front_list: List[List[float]] = []
    wheel_rear_list: List[List[float]] = []

    vehicle_boxes_xyxy: List[np.ndarray] = []
    wheel_front_boxes_xyxy: List[np.ndarray] = []
    wheel_rear_boxes_xyxy: List[np.ndarray] = []

    template_front: np.ndarray | None = None
    template_rear: np.ndarray | None = None
    prev_car_patch: np.ndarray | None = None

    for idx in tqdm(
        range(n_frames), desc="Pass 2/2: Track wheels & body points", unit="frame"
    ):
        ret, frame = cap.read()
        if not ret:
            break

        det = frame_dets[idx]
        car_box = det.vehicle_box

        if car_box is None and prev_car_patch is not None and prev_car_patch.size > 0:
            tl, br = match_template_box(frame, prev_car_patch)
            car_box = (tl, br)

        if car_box is None:
            body_front_list.append([np.nan, np.nan])
            body_rear_list.append([np.nan, np.nan])
            wheel_front_list.append([np.nan, np.nan])
            wheel_rear_list.append([np.nan, np.nan])

            vehicle_boxes_xyxy.append(_box_to_xyxy_or_nan(None))
            wheel_front_boxes_xyxy.append(_box_to_xyxy_or_nan(None))
            wheel_rear_boxes_xyxy.append(_box_to_xyxy_or_nan(None))
            continue

        (cx1, cy1), (cx2, cy2) = car_box
        car_img = frame[cy1:cy2, cx1:cx2]
        if car_img.size == 0:
            body_front_list.append([np.nan, np.nan])
            body_rear_list.append([np.nan, np.nan])
            wheel_front_list.append([np.nan, np.nan])
            wheel_rear_list.append([np.nan, np.nan])

            vehicle_boxes_xyxy.append(_box_to_xyxy_or_nan(car_box))
            wheel_front_boxes_xyxy.append(_box_to_xyxy_or_nan(None))
            wheel_rear_boxes_xyxy.append(_box_to_xyxy_or_nan(None))
            continue

        prev_car_patch = car_img.copy()

        mid_x = car_img.shape[1] // 2

        car_front = car_img[:, :mid_x, :]
        car_rear = car_img[:, mid_x:, :]

        top_left_car_front = (cx1, cy1)
        top_left_car_rear = (cx1 + mid_x, cy1)

        if template_front is None or template_rear is None:
            if det.wheel_front_box is None or det.wheel_rear_box is None:
                body_front_list.append([np.nan, np.nan])
                body_rear_list.append([np.nan, np.nan])
                wheel_front_list.append([np.nan, np.nan])
                wheel_rear_list.append([np.nan, np.nan])

                vehicle_boxes_xyxy.append(_box_to_xyxy_or_nan(car_box))
                wheel_front_boxes_xyxy.append(_box_to_xyxy_or_nan(det.wheel_front_box))
                wheel_rear_boxes_xyxy.append(_box_to_xyxy_or_nan(det.wheel_rear_box))
                continue

            (wf_tl, wf_br) = det.wheel_front_box
            (wr_tl, wr_br) = det.wheel_rear_box

            wf_patch = frame[wf_tl[1] : wf_br[1], wf_tl[0] : wf_br[0]]
            wr_patch = frame[wr_tl[1] : wr_br[1], wr_tl[0] : wr_br[0]]

            if wf_patch.size == 0 or wr_patch.size == 0:
                body_front_list.append([np.nan, np.nan])
                body_rear_list.append([np.nan, np.nan])
                wheel_front_list.append([np.nan, np.nan])
                wheel_rear_list.append([np.nan, np.nan])

                vehicle_boxes_xyxy.append(_box_to_xyxy_or_nan(car_box))
                wheel_front_boxes_xyxy.append(_box_to_xyxy_or_nan(det.wheel_front_box))
                wheel_rear_boxes_xyxy.append(_box_to_xyxy_or_nan(det.wheel_rear_box))
                continue

            template_front = wf_patch.copy()
            template_rear = wr_patch.copy()

        wf_hint_local: Box | None = None
        wr_hint_local: Box | None = None

        if det.wheel_front_box is not None:
            (wf_tl, wf_br) = det.wheel_front_box
            wf_hint_local = (
                (wf_tl[0] - top_left_car_front[0], wf_tl[1] - top_left_car_front[1]),
                (wf_br[0] - top_left_car_front[0], wf_br[1] - top_left_car_front[1]),
            )

        if det.wheel_rear_box is not None:
            (wr_tl, wr_br) = det.wheel_rear_box
            wr_hint_local = (
                (wr_tl[0] - top_left_car_rear[0], wr_tl[1] - top_left_car_rear[1]),
                (wr_br[0] - top_left_car_rear[0], wr_br[1] - top_left_car_rear[1]),
            )

        wf_center_local, wf_box_local, bf_local = find_body_point_and_wheel(
            car_front,
            template_front,
            majority_label,
            use_subpixel=use_subpixel,
            wheel_box_hint=wf_hint_local,
        )

        wr_center_local, wr_box_local, br_local = find_body_point_and_wheel(
            car_rear,
            template_rear,
            majority_label,
            use_subpixel=use_subpixel,
            wheel_box_hint=wr_hint_local,
        )

        (wfx1, wfy1), (wfx2, wfy2) = wf_box_local
        wf_patch_new = car_front[wfy1:wfy2, wfx1:wfx2]
        if wf_patch_new.size > 0:
            template_front = wf_patch_new.copy()

        (wrx1, wry1), (wrx2, wry2) = wr_box_local
        wr_patch_new = car_rear[wry1:wry2, wrx1:wrx2]
        if wr_patch_new.size > 0:
            template_rear = wr_patch_new.copy()

        wf_center_abs = np.array(
            [
                top_left_car_front[0] + wf_center_local[0],
                top_left_car_front[1] + wf_center_local[1],
            ],
            dtype=np.float32,
        )
        wr_center_abs = np.array(
            [
                top_left_car_rear[0] + wr_center_local[0],
                top_left_car_rear[1] + wr_center_local[1],
            ],
            dtype=np.float32,
        )
        bf_abs = np.array(
            [top_left_car_front[0] + bf_local[0], top_left_car_front[1] + bf_local[1]],
            dtype=np.float32,
        )
        br_abs = np.array(
            [top_left_car_rear[0] + br_local[0], top_left_car_rear[1] + br_local[1]],
            dtype=np.float32,
        )

        # Wheel boxes to full-image coordinates
        wf_box_abs: Box = (
            (
                top_left_car_front[0] + wf_box_local[0][0],
                top_left_car_front[1] + wf_box_local[0][1],
            ),
            (
                top_left_car_front[0] + wf_box_local[1][0],
                top_left_car_front[1] + wf_box_local[1][1],
            ),
        )
        wr_box_abs: Box = (
            (
                top_left_car_rear[0] + wr_box_local[0][0],
                top_left_car_rear[1] + wr_box_local[0][1],
            ),
            (
                top_left_car_rear[0] + wr_box_local[1][0],
                top_left_car_rear[1] + wr_box_local[1][1],
            ),
        )

        shift_vec = np.array([det.shift.dx, det.shift.dy], dtype=np.float32)

        wheel_front_stab = wf_center_abs - shift_vec
        wheel_rear_stab = wr_center_abs - shift_vec
        body_front_stab = bf_abs - shift_vec
        body_rear_stab = br_abs - shift_vec

        # Apply shift to boxes as well (stabilized boxes)
        wf_box_abs_xyxy = _box_to_xyxy_or_nan(wf_box_abs)
        wr_box_abs_xyxy = _box_to_xyxy_or_nan(wr_box_abs)
        veh_box_xyxy = _box_to_xyxy_or_nan(car_box)

        wf_box_abs_xyxy[:2] -= shift_vec
        wf_box_abs_xyxy[2:] -= shift_vec
        wr_box_abs_xyxy[:2] -= shift_vec
        wr_box_abs_xyxy[2:] -= shift_vec
        veh_box_xyxy[:2] -= shift_vec
        veh_box_xyxy[2:] -= shift_vec

        wheel_front_list.append(
            [float(wheel_front_stab[0]), float(wheel_front_stab[1])]
        )
        wheel_rear_list.append([float(wheel_rear_stab[0]), float(wheel_rear_stab[1])])
        body_front_list.append([float(body_front_stab[0]), float(body_front_stab[1])])
        body_rear_list.append([float(body_rear_stab[0]), float(body_rear_stab[1])])

        vehicle_boxes_xyxy.append(veh_box_xyxy.astype(np.float32))
        wheel_front_boxes_xyxy.append(wf_box_abs_xyxy.astype(np.float32))
        wheel_rear_boxes_xyxy.append(wr_box_abs_xyxy.astype(np.float32))

    cap.release()

    return {
        "body_front": np.asarray(body_front_list, dtype=np.float32),
        "body_rear": np.asarray(body_rear_list, dtype=np.float32),
        "wheel_front": np.asarray(wheel_front_list, dtype=np.float32),
        "wheel_rear": np.asarray(wheel_rear_list, dtype=np.float32),
        "vehicle_box": np.asarray(vehicle_boxes_xyxy, dtype=np.float32),
        "wheel_front_box": np.asarray(wheel_front_boxes_xyxy, dtype=np.float32),
        "wheel_rear_box": np.asarray(wheel_rear_boxes_xyxy, dtype=np.float32),
    }
