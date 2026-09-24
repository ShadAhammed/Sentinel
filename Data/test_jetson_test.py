"""Tests for the shared YOLO and CNN label map."""

import unittest

import jetson_test


class JetsonLabelTests(unittest.TestCase):
    def test_spelling_and_vehicle_names_use_cnn_labels(self) -> None:
        self.assertEqual(jetson_test.cnn_label("Artilary"), "Artillery")
        self.assertEqual(jetson_test.cnn_label("Tank"), "military_vehicle")
        self.assertEqual(jetson_test.cnn_label("tank"), "military_vehicle")
        self.assertEqual(jetson_test.cnn_label("armoured personnel carrier"), "military_vehicle")
        self.assertEqual(jetson_test.cnn_label("Soldier"), "Soldier")
        # MV spells the camouflage class in lower case.
        self.assertEqual(jetson_test.cnn_label("soldier"), "camouflage_soldier")

    def test_dropped_yolo_names_are_not_cnn_labels(self) -> None:
        self.assertIsNone(jetson_test.cnn_label("Vehicle"))
        self.assertIsNone(jetson_test.cnn_label("weapon"))
        self.assertIsNone(jetson_test.cnn_label("civilian"))
        self.assertIsNone(jetson_test.cnn_label("civilian_vehicle"))

    def test_label_files_use_the_dataset_meaning_of_soldier(self) -> None:
        self.assertEqual(jetson_test.source_label("MV", "soldier"), "camouflage_soldier")
        self.assertEqual(jetson_test.source_label("military_object_dataset", "soldier"), "Soldier")
        self.assertEqual(jetson_test.source_label("KIIT-MiTA", "Artilary"), "Artillery")
        self.assertEqual(jetson_test.source_label("military_object_dataset", "military_tank"), "military_vehicle")
        self.assertIsNone(jetson_test.source_label("KIIT-MiTA", "Vehicle"))

    def test_shared_list_has_the_ten_cnn_labels(self) -> None:
        self.assertEqual(len(jetson_test.CNN_LABELS), 10)
        for name in jetson_test.YOLO_TO_CNN.values():
            self.assertIn(name, jetson_test.CNN_LABELS)


if __name__ == "__main__":
    unittest.main()
