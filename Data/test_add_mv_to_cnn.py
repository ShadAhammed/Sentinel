"""Tests for the MV crop labels added to CNN-Data."""

import unittest

import add_mv_to_cnn


class AddMvTests(unittest.TestCase):
    def test_mv_ids_map_onto_two_cnn_labels(self) -> None:
        self.assertEqual(add_mv_to_cnn.mv_box_label(0), "military_vehicle")
        self.assertEqual(add_mv_to_cnn.mv_box_label(2), "military_vehicle")
        self.assertEqual(add_mv_to_cnn.mv_box_label(1), "camouflage_soldier")
        self.assertIsNone(add_mv_to_cnn.mv_box_label(3))


if __name__ == "__main__":
    unittest.main()
