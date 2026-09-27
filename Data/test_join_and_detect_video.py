"""Tests for box overlap and the small-crop gate."""

import unittest

import numpy as np

import join_and_detect_video


class DetectTests(unittest.TestCase):
    def test_overlap_keeps_the_stronger_box(self) -> None:
        boxes = [
            {"xyxy": [0, 0, 100, 100], "confidence": 0.4, "label": "Soldier"},
            {"xyxy": [10, 10, 90, 90], "confidence": 0.9, "label": "military_vehicle"},
        ]
        kept = join_and_detect_video.keep_best_boxes(boxes)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["label"], "military_vehicle")

    def test_separate_boxes_are_both_kept(self) -> None:
        boxes = [
            {"xyxy": [0, 0, 40, 40], "confidence": 0.5, "label": "Soldier"},
            {"xyxy": [200, 200, 280, 280], "confidence": 0.8, "label": "Radar"},
        ]
        kept = join_and_detect_video.keep_best_boxes(boxes)
        self.assertEqual(len(kept), 2)

    def test_crop_under_32_pixels_is_skipped(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        found = join_and_detect_video.cnn_prediction_for_crop(None, frame, [0, 0, 20, 40], "cpu")
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
