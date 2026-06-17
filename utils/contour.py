import numpy as np
import cv2
from typing import Dict, List, Tuple, Any

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.preprocess import binarize


class Contour:
    __slots__ = ("pattern_contours", "width")

    def __init__(self, wsi_width: int) -> None:
        self.pattern_contours: Dict[str, List[np.ndarray]] = {
            "pattern3": [],
            "pattern4": [],
            "pattern5": [],
        }
        self.width: int = int(wsi_width)

    def stack(self, *args: Any, **kwargs: Any) -> None:
        if args and len(args) == 3 and not kwargs:
            pattern_type, img, offset = args  # why: keep backward compatibility with caller
        else:
            pattern_type = kwargs["pattern_type"]
            img = kwargs["img"]
            offset = kwargs["offset"]

        binary = binarize(img)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=tuple(offset)
        )
        self.pattern_contours[pattern_type].extend(list(contours))

    def formatting(self, contour: np.ndarray) -> str:
        contour = contour.astype(np.float32)
        contour /= self.width
        contour = contour.reshape(-1).tolist()
        return ", ".join(str(c) for c in contour)

    def to_json(self) -> Dict[str, Dict[str, str]]:
        pattern_json: Dict[str, Dict[str, str]] = {}
        for pattern, contours in self.pattern_contours.items():
            for idx in range(len(contours)):
                contours[idx] = self.formatting(contours[idx])  # type: ignore[assignment]
            pattern_json[pattern] = {str(idx): contours[idx] for idx in range(len(contours))}
        return pattern_json