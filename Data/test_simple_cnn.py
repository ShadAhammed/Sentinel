"""Tests for the small CNN label map, caps, and network shape."""

import random
import unittest

import torch

import simple_cnn


class SimpleCnnTests(unittest.TestCase):
    def test_mv_labels_fold_into_three_existing_classes(self) -> None:
        self.assertIsNone(simple_cnn.mv_final_label("air-fighter"))
        self.assertIsNone(simple_cnn.mv_final_label("bomber"))
        self.assertEqual(simple_cnn.mv_final_label("armoured personnel carrier"), "military_vehicle")
        self.assertEqual(simple_cnn.mv_final_label("tank"), "military_vehicle")
        self.assertEqual(simple_cnn.mv_final_label("soldier"), "camouflage_soldier")
        self.assertIsNone(simple_cnn.mv_final_label("butterfly"))

    def test_cap_stops_when_enough_rows_pass(self) -> None:
        rows = [{"key": f"{index:02d}", "ok": index % 2 == 0} for index in range(10)]
        chosen = simple_cnn.take_rows(rows, 3, random.Random(42), lambda row: row["ok"])
        self.assertEqual(len(chosen), 3)
        self.assertTrue(all(row["ok"] for row in chosen))

    def test_network_is_small_and_outputs_ten_classes(self) -> None:
        model = simple_cnn.SmallCNN(len(simple_cnn.CLASS_NAMES))
        parameter_count = simple_cnn.count_parameters(model)
        self.assertLess(parameter_count, 200_000)
        self.assertEqual(simple_cnn.IMAGE_SIZE, 96)
        output = model(torch.zeros(2, 3, 96, 96))
        self.assertEqual(tuple(output.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
