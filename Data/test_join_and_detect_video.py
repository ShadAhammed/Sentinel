"""Tests for joining the drone clips."""

import unittest

import join_and_detect_video


class JoinVideoTests(unittest.TestCase):
    def test_crossfade_starts_before_each_clip_ends(self) -> None:
        offsets = join_and_detect_video.fade_offsets([10.0, 10.0, 6.0, 6.0], 0.8)
        self.assertEqual(len(offsets), 3)
        self.assertAlmostEqual(offsets[0], 9.2)
        self.assertAlmostEqual(offsets[1], 18.4)
        self.assertAlmostEqual(offsets[2], 23.6)

    def test_overlap_keeps_the_stronger_box(self) -> None:
        boxes = [
            {"xyxy": [0, 0, 100, 100], "confidence": 0.4, "label": "Soldier"},
            {"xyxy": [10, 10, 90, 90], "confidence": 0.9, "label": "military_vehicle"},
        ]
        kept = join_and_detect_video.keep_best_boxes(boxes)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["label"], "military_vehicle")


if __name__ == "__main__":
    unittest.main()
