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
from typing import Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

Point = Tuple[float, float]
IntPoint = Tuple[int, int]
Box = Tuple[IntPoint, IntPoint]


def relu(x: int) -> int:
    """
    Rectified linear unit for integers.

    Args:
        x:
            Input integer.

    Returns:
        max(x, 0)
    """
    return x if x >= 0 else 0


def error_warning(img: np.ndarray) -> bool:
    """
    Detect potentially corrupted frames using feature-point count.

    Args:
        img:
            Input frame (BGR).

    Returns:
        True if the frame is considered unreliable, otherwise False.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    if corners is None or len(corners) < 30:
        print("Video data seems corrupted (too few feature points).")
        return True
    return False


@dataclass
class BackgroundShift:
    """
    Background translation estimate between consecutive frames.

    Attributes:
        dx:
            Translation in x direction (pixels).
        dy:
            Translation in y direction (pixels).
    """

    dx: float
    dy: float


def _clip_box(box: Box, w: int, h: int) -> Box:
    """
    Clip a box to image bounds.

    Args:
        box:
            Input box.
        w:
            Image width.
        h:
            Image height.

    Returns:
        Clipped box.
    """
    (x1, y1), (x2, y2) = box
    x1c = int(np.clip(x1, 0, w - 1))
    y1c = int(np.clip(y1, 0, h - 1))
    x2c = int(np.clip(x2, 1, w))
    y2c = int(np.clip(y2, 1, h))
    if x2c <= x1c:
        x2c = min(w, x1c + 1)
    if y2c <= y1c:
        y2c = min(h, y1c + 1)
    return (x1c, y1c), (x2c, y2c)


def _enlarge_box(box: Box, scale: float, w: int, h: int) -> Box:
    """
    Enlarge a box around its center, then clip to image bounds.

    Args:
        box:
            Input box.
        scale:
            Scale factor (e.g., 1.1).
        w:
            Image width.
        h:
            Image height.

    Returns:
        Enlarged and clipped box.
    """
    (x1, y1), (x2, y2) = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = int(round(cx - bw / 2))
    nx2 = int(round(cx + bw / 2))
    ny1 = int(round(cy - bh / 2))
    ny2 = int(round(cy + bh / 2))
    return _clip_box(((nx1, ny1), (nx2, ny2)), w, h)


def _validate_car_and_template(
    car: np.ndarray | None, template: np.ndarray | None
) -> bool:
    """
    Validate inputs for template-based processing.

    Args:
        car:
            Car image.
        template:
            Template image.

    Returns:
        True if valid, otherwise False.
    """
    if car is None or template is None:
        return False
    if car.size == 0 or template.size == 0:
        return False
    if car.ndim != 3 or template.ndim != 3:
        return False
    if car.shape[2] != 3 or template.shape[2] != 3:
        return False
    return True


def _ensure_template_smaller(car: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Ensure the template fits inside the search image by resizing if necessary.

    Args:
        car:
            Search image.
        template:
            Template image.

    Returns:
        Template resized if needed.
    """
    h_car, w_car, _ = car.shape
    h_t, w_t, _ = template.shape
    if h_t < h_car and w_t < w_car:
        return template

    scale = min(
        (w_car - 1) / max(w_t, 1),
        (h_car - 1) / max(h_t, 1),
    )
    new_w = max(2, int(w_t * scale))
    new_h = max(2, int(h_t * scale))
    return cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)


def match_template_box(
    search_img: np.ndarray, template: np.ndarray
) -> Tuple[IntPoint, IntPoint]:
    """
    Locate a template in a search image using normalized cross-correlation.

    Args:
        search_img:
            Search image.
        template:
            Template image.

    Returns:
        (top_left, bottom_right) box in search_img coordinates.
    """
    if (
        search_img is None
        or template is None
        or search_img.size == 0
        or template.size == 0
    ):
        return (0, 0), (0, 0)

    hS, wS = search_img.shape[:2]
    hT, wT = template.shape[:2]
    if hT >= hS or wT >= wS:
        tmpl = _ensure_template_smaller(search_img, template)
    else:
        tmpl = template

    res = cv2.matchTemplate(search_img, tmpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    tl = (int(max_loc[0]), int(max_loc[1]))
    br = (int(max_loc[0] + tmpl.shape[1]), int(max_loc[1] + tmpl.shape[0]))
    tl, br = _clip_box((tl, br), wS, hS)
    return tl, br


def _refine_wheel_box_by_threshold(car: np.ndarray, rough_box: Box, label: str) -> Box:
    """
    Refine a rough wheel bounding box using thresholding and contour selection.

    Args:
        car:
            Car image (BGR).
        rough_box:
            Rough wheel box in car coordinates.
        label:
            Vehicle label (e.g., car, bus, truck).

    Returns:
        Refined wheel box in car coordinates.
    """
    h, w = car.shape[:2]
    rough_box = _clip_box(rough_box, w, h)
    enlarged = _enlarge_box(rough_box, scale=1.1, w=w, h=h)

    (x1, y1), (x2, y2) = enlarged
    crop = car[y1:y2, x1:x2]
    if crop.size == 0:
        return rough_box

    if label in ("bus", "truck"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = clahe.apply(g)
        crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    crop_f = cv2.bilateralFilter(crop, 9, 75, 75)
    crop_g = cv2.cvtColor(crop_f, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(crop_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) <= 1:
        return enlarged

    candidates = contours[:-1] if len(contours) > 1 else contours
    areas = np.array([cv2.contourArea(c) for c in candidates], dtype=float)
    if areas.size == 0:
        return enlarged

    idx = int(np.argmax(areas))
    cnt = candidates[idx]
    x, y, bw, bh = cv2.boundingRect(cnt)

    refined = ((x1 + x, y1 + y), (x1 + x + bw, y1 + y + bh))
    refined = _clip_box(refined, w, h)
    return refined


def _crop_body_strip(
    car: np.ndarray, wheel_box: Box, label: str
) -> tuple[np.ndarray, int, int]:
    """
    Crop a vertical strip above the wheel region for body-point estimation.

    Args:
        car:
            Car image (BGR).
        wheel_box:
            Wheel box in car coordinates.
        label:
            Vehicle label.

    Returns:
        (strip_image, left_x_offset, top_y_offset)
    """
    (x1, y1), (x2, y2) = wheel_box
    h_car, w_car, _ = car.shape

    crop_loc = int(h_car * 4 / 10) if label == "car" else int(h_car * 5 / 10)

    height_wheel = y2 - y1
    top_boundary = y1 - int(height_wheel / 10)

    l_crop, r_crop = x1, x2

    if crop_loc >= top_boundary or l_crop >= r_crop:
        return np.empty((0, 0, 3), dtype=car.dtype), l_crop, crop_loc

    strip = car[crop_loc:top_boundary, l_crop:r_crop, :]
    return strip, l_crop, crop_loc


def _find_vertical_peak(
    filtered_gray: np.ndarray, col: int, use_subpixel: bool
) -> float:
    """
    Find a peak along a selected column in a filtered grayscale image.

    Args:
        filtered_gray:
            Filtered grayscale image.
        col:
            Column index.
        use_subpixel:
            Whether to apply quadratic subpixel refinement.

    Returns:
        Estimated peak row position (float).
    """
    col = int(np.clip(col, 0, filtered_gray.shape[1] - 1))
    y_line = filtered_gray[:, col]

    peaks, _ = find_peaks(y_line, distance=10, prominence=30)
    if len(peaks) == 0:
        peaks, _ = find_peaks(y_line, distance=10, prominence=10)
    if len(peaks) == 0:
        return float(filtered_gray.shape[0] // 2)

    top_y = peaks[0] if len(peaks) == 1 else peaks[-1]

    if not use_subpixel:
        return float(top_y)

    y0 = max(top_y - 1, 0)
    y1 = min(top_y + 1, filtered_gray.shape[0] - 1)
    if y1 - y0 < 2:
        return float(top_y)

    y_idx = np.arange(y0, y1 + 1)
    vals = filtered_gray[y0 : y1 + 1, col]
    coeffs = np.polyfit(y_idx, vals, 2)
    a, b, _ = coeffs
    if a == 0:
        return float(top_y)
    return float(-b / (2 * a))


def find_body_point_and_wheel(
    car: np.ndarray,
    template_wheel: np.ndarray,
    label: str,
    use_subpixel: bool = True,
    wheel_box_hint: Box | None = None,
) -> tuple[Point, Box, Point]:
    """
    Estimate wheel center, wheel bounding box, and a body point above the wheel.

    Args:
        car:
            Car image for either front-half or rear-half (BGR).
        template_wheel:
            Wheel template (BGR) used when wheel_box_hint is not available.
        label:
            Vehicle label.
        use_subpixel:
            Whether to apply subpixel refinement for the body point.
        wheel_box_hint:
            Optional wheel box hint in car coordinates. If provided, template matching
            is not used for wheel localization.

    Returns:
        wheel_center:
            (x, y) wheel center in car coordinates.
        wheel_box:
            Wheel box ((x1,y1),(x2,y2)) in car coordinates.
        body_point:
            (x, y) body point in car coordinates.
    """
    if not _validate_car_and_template(car, template_wheel):
        return (np.nan, np.nan), ((0, 0), (0, 0)), (np.nan, np.nan)

    h, w = car.shape[:2]

    if wheel_box_hint is not None:
        rough = _clip_box(wheel_box_hint, w, h)
        wheel_box = _refine_wheel_box_by_threshold(car, rough, label)
    else:
        template = _ensure_template_smaller(car, template_wheel)
        res = cv2.matchTemplate(car, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        tl = (int(max_loc[0]), int(max_loc[1]))
        br = (int(max_loc[0] + template.shape[1]), int(max_loc[1] + template.shape[0]))
        rough = _clip_box((tl, br), w, h)
        wheel_box = _refine_wheel_box_by_threshold(car, rough, label)

    (wx1, wy1), (wx2, wy2) = wheel_box
    wheel_center = (float((wx1 + wx2) / 2), float((wy1 + wy2) / 2))

    strip, l_crop, crop_loc = _crop_body_strip(car, wheel_box, label)
    if strip.size == 0 or strip.shape[0] < 3 or strip.shape[1] < 3:
        body_point = (float(l_crop), float(crop_loc))
        return wheel_center, wheel_box, body_point

    strip_filtered = cv2.bilateralFilter(strip, 10, 75, 75)

    sx = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = (1 / 8) * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    s_x = -sx
    s_y = -sy

    sharp1 = cv2.filter2D(strip_filtered, -1, sx)
    sharp2 = cv2.filter2D(strip_filtered, -1, sy)
    sharp3 = cv2.filter2D(strip_filtered, -1, s_x)
    sharp4 = cv2.filter2D(strip_filtered, -1, s_y)

    filtered = sharp1 + sharp2 + sharp3 + sharp4
    filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    center_col = int((wx1 + wx2) / 2) - l_crop
    top_y = _find_vertical_peak(filtered_gray, center_col, use_subpixel)
    body_point = (float(l_crop + center_col), float(crop_loc + top_y))

    return wheel_center, wheel_box, body_point


def estimate_background_shift(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    roi_ratio: int = 10,
) -> BackgroundShift:
    """
    Estimate background translation between two consecutive frames.

    Args:
        prev_frame:
            Previous frame (BGR).
        current_frame:
            Current frame (BGR).
        roi_ratio:
            Defines a top-right ROI size as (W/roi_ratio, H/roi_ratio).

    Returns:
        BackgroundShift(dx, dy)
    """
    h, w = current_frame.shape[:2]
    x0 = int(w - w / roi_ratio)
    y0 = 0
    x1 = w
    y1 = int(h / roi_ratio)

    roi_prev = prev_frame[y0:y1, x0:x1]
    roi_curr = current_frame[y0:y1, x0:x1]

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(roi_prev, None)
    kp2, desc2 = sift.detectAndCompute(roi_curr, None)

    if desc1 is None or desc2 is None:
        return BackgroundShift(dx=0.0, dy=0.0)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)

    if len(good) < 4:
        return BackgroundShift(dx=0.0, dy=0.0)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    A, _ = cv2.estimateAffine2D(pts1, pts2)
    if A is None:
        return BackgroundShift(dx=0.0, dy=0.0)

    return BackgroundShift(dx=float(A[0, 2]), dy=float(A[1, 2]))
