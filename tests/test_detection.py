"""Tests for YOLO box deduplication, crop gating, and EfficientNet hit logic."""

import unittest

import numpy as np

from src.detection import (
    confidence_percent,
    efficientnet_hits,
    final_label,
    iou,
    keep_best_boxes,
    cnn_prediction_for_crop,
)


class IouTests(unittest.TestCase):

    def test_identical_boxes_have_iou_one(self) -> None:
        box = [0, 0, 100, 100]
        self.assertAlmostEqual(iou(box, box), 1.0)

    def test_non_overlapping_boxes_have_iou_zero(self) -> None:
        self.assertEqual(iou([0, 0, 50, 50], [100, 100, 200, 200]), 0.0)

    def test_partial_overlap_is_between_zero_and_one(self) -> None:
        result = iou([0, 0, 100, 100], [50, 50, 150, 150])
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)


class KeepBestBoxesTests(unittest.TestCase):

    def test_overlapping_boxes_keep_the_stronger_one(self) -> None:
        boxes = [
            {"xyxy": [0, 0, 100, 100], "confidence": 0.4, "label": "Soldier"},
            {"xyxy": [10, 10, 90, 90],  "confidence": 0.9, "label": "military_vehicle"},
        ]
        kept = keep_best_boxes(boxes)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["label"], "military_vehicle")

    def test_separate_boxes_are_both_kept(self) -> None:
        boxes = [
            {"xyxy": [0, 0, 40, 40],   "confidence": 0.5, "label": "Soldier"},
            {"xyxy": [200, 200, 280, 280], "confidence": 0.8, "label": "Radar"},
        ]
        kept = keep_best_boxes(boxes)
        self.assertEqual(len(kept), 2)

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(keep_best_boxes([]), [])


class CropGateTests(unittest.TestCase):

    def test_crop_under_32_pixels_wide_is_skipped(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cnn_prediction_for_crop(None, frame, [0, 0, 20, 40], "cpu")
        self.assertIsNone(result)

    def test_crop_under_32_pixels_tall_is_skipped(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cnn_prediction_for_crop(None, frame, [0, 0, 40, 20], "cpu")
        self.assertIsNone(result)


class FinalLabelTests(unittest.TestCase):

    def test_cnn_name_wins_when_present(self) -> None:
        self.assertEqual(final_label("Soldier", "camouflage_soldier"), "camouflage_soldier")
        self.assertEqual(final_label("Soldier", "Soldier"), "Soldier")

    def test_yolo_name_used_when_cnn_is_none(self) -> None:
        self.assertEqual(final_label("Soldier", None), "Soldier")


class EfficientnetHitsTests(unittest.TestCase):

    def test_boxes_without_cnn_name_are_dropped(self) -> None:
        candidates = [
            {"xyxy": [0, 0, 40, 40], "cnn_name": None, "cnn_confidence": None},
            {"xyxy": [100, 100, 200, 200], "cnn_name": "Radar", "cnn_confidence": 0.9},
        ]
        kept = efficientnet_hits(candidates)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["cnn_name"], "Radar")

    def test_overlapping_named_boxes_keep_the_highest_cnn_score(self) -> None:
        candidates = [
            {"xyxy": [0, 0, 100, 100], "cnn_name": "Soldier", "cnn_confidence": 0.4},
            {"xyxy": [10, 10, 110, 110], "cnn_name": "military_vehicle", "cnn_confidence": 0.9},
        ]
        kept = efficientnet_hits(candidates)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["cnn_name"], "military_vehicle")

    def test_non_overlapping_named_boxes_are_both_kept(self) -> None:
        candidates = [
            {"xyxy": [0, 0, 100, 100],     "cnn_name": "Soldier",  "cnn_confidence": 0.7},
            {"xyxy": [400, 400, 500, 500],  "cnn_name": "Soldier",  "cnn_confidence": 0.7},
        ]
        kept = efficientnet_hits(candidates)
        self.assertEqual(len(kept), 2)


class ConfidencePercentTests(unittest.TestCase):

    def test_rounds_to_nearest_whole_percent(self) -> None:
        self.assertEqual(confidence_percent(0.876), "88%")
        self.assertEqual(confidence_percent(0.5),   "50%")
        self.assertEqual(confidence_percent(1.0),  "100%")
        self.assertEqual(confidence_percent(0.0),    "0%")


if __name__ == "__main__":
    unittest.main()
