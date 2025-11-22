# -----------------------------------------------------------------------------
# Copyright (c) 2025 Atsushi ISHII
#
# This file is part of Vision-based modal identification and weight estimation of vehicles.
#
# Licensed under the MIT License. See the LICENSE file in the project root
# for full license information.
# -----------------------------------------------------------------------------

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple

sys.path.append("./yolov5")  # keep this so yolov5 repo is importable

import warnings

import cv2
import numpy as np
import torch

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
)

IntPoint = Tuple[int, int]
Box = Tuple[IntPoint, IntPoint]


@dataclass
class DetectionResult:
    """
    Simple container for YOLO detection outputs.

    Attributes:
        locs:
            List of bounding boxes ((x1, y1), (x2, y2)) in image coordinates.
        labels:
            List of class names (e.g., 'car', 'bus').
            May be None for wheel detector.
        scores:
            List of confidence scores.
    """

    locs: List[Box] | None
    labels: List[str] | None
    scores: List[float] | None


class _BaseDetector:
    """
    Base class for YOLOv5-based detectors.

    This wraps your existing YOLOv5 weights and inference code.
    """

    def __init__(self, img_size: int, weights: str, conf_thres: float):
        self.imgsz = img_size
        self.weights = weights
        self.conf_thres = conf_thres

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        set_logging()
        warnings.simplefilter("ignore")

        # Load model
        model = attempt_load(weights, map_location=device)
        self.model = model

        # Adjust image size to multiple of model stride
        self.imgsz = check_img_size(self.imgsz, s=model.stride.max())

        # Half precision if CUDA
        self.half = device.type != "cpu"
        if self.half:
            model.half()

        # Class names
        self.names = model.module.names if hasattr(model, "module") else model.names

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Resize image to the inference size and convert to tensor.
        """
        imgsz = self.imgsz
        h0, w0 = image.shape[:2]

        # Keep aspect ratio, make width multiple of 32 (similar to your logic)
        w_modif = int(w0 * imgsz / h0)
        q32 = w_modif // 32
        mod32 = w_modif % 32
        if mod32 < 16:
            w1 = q32 * 32
        else:
            w1 = q32 * 32 + 32

        img = cv2.resize(image, (int(w1), imgsz))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
        img = np.ascontiguousarray(img)

        img_t = torch.from_numpy(img).to(self.device)
        img_t = img_t.half() if self.half else img_t.float()
        img_t /= 255.0

        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)
        return img_t

    @torch.inference_mode()
    def _infer(self, image: np.ndarray, conf_thres: float) -> DetectionResult:
        """
        Run YOLOv5 on a single BGR image and return raw detection outputs.

        Args:
            image:
                BGR image in HxWx3 format.
            conf_thres:
                Confidence threshold.

        Returns:
            DetectionResult with bounding boxes in the original image coordinates.
        """
        img_t = self._preprocess(image)
        model = self.model
        pred = model(img_t, augment=False)[0]
        iou_thres = 0.5

        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
        )

        # Process detections
        locs: List[Box] = []
        labels: List[str] = []
        scores: List[float] = []

        h_img, w_img = image.shape[:2]
        for det in pred:
            if det is not None and len(det):
                # Rescale from network input size to original image size
                det[:, :4] = scale_coords(
                    img_t.shape[2:], det[:, :4], (h_img, w_img, 3)
                ).round()

                for *xyxy, conf, cls in reversed(det):
                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())

                    top_left = (x1, y1)
                    bottom_right = (x2, y2)
                    label = self.names[int(cls)]
                    locs.append((top_left, bottom_right))
                    labels.append(label)
                    scores.append(float(conf.item()))

        if not locs:
            return DetectionResult(locs=None, labels=None, scores=None)

        return DetectionResult(locs=locs, labels=labels, scores=scores)


class Detector_car(_BaseDetector):
    """
    YOLOv5-based vehicle detector (car/bus/truck).
    """

    def __init__(self):
        super().__init__(img_size=416, weights="weights/yolov5_car.pt", conf_thres=0.5)

    def detect_car(self, image: np.ndarray) -> DetectionResult:
        """
        Detect cars/buses/trucks in the image and return only relevant objects.

        Args:
            image:
                BGR image.

        Returns:
            DetectionResult with bounding boxes of vehicles.
        """
        result = self._infer(image, conf_thres=self.conf_thres)
        if result.locs is None:
            return result

        filtered_locs: List[Box] = []
        filtered_labels: List[str] = []
        filtered_scores: List[float] = []

        for loc, label, score in zip(result.locs, result.labels, result.scores):
            # Only keep 'car', 'bus', 'truck' etc. (class id 2 in original code
            # is model-specific; here we filter by semantic label)
            if label in ("car", "bus", "truck"):
                filtered_locs.append(loc)
                filtered_labels.append(label)
                filtered_scores.append(score)

        if not filtered_locs:
            return DetectionResult(locs=None, labels=None, scores=None)

        return DetectionResult(
            locs=filtered_locs,
            labels=filtered_labels,
            scores=filtered_scores,
        )


class Detector_wheels(_BaseDetector):
    """
    YOLOv5-based wheel detector.

    The wheel model is assumed to have a single 'wheel' class.
    """

    def __init__(self):
        super().__init__(
            img_size=320, weights="weights/yolov5_wheel.pt", conf_thres=0.8
        )

    def detect_wheels(self, image: np.ndarray) -> DetectionResult:
        """
        Detect wheels in the given car crop.

        Args:
            image:
                BGR cropped image of a vehicle.

        Returns:
            DetectionResult with wheel bounding boxes.
        """
        result = self._infer(image, conf_thres=self.conf_thres)
        if result.locs is None:
            return DetectionResult(locs=None, labels=None, scores=None)
        # Ignore labels for wheels (single class)
        return DetectionResult(locs=result.locs, labels=None, scores=result.scores)
