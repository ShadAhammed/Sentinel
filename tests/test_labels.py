"""Tests for the CNN label list and the YOLO-to-CNN name mapping."""

import unittest

from sentinel.labels import CNN_LABELS, EFFNET_PATH, VIDEO_PATH, YOLO_FILES, YOLO_TO_CNN, cnn_label


class LabelMapTests(unittest.TestCase):

    def test_misspelled_and_renamed_yolo_classes_map_correctly(self) -> None:
        # KIIT-MiTA spells Artillery with one 'l'. Both spellings must map.
        self.assertEqual(cnn_label("Artilary"), "Artillery")
        self.assertEqual(cnn_label("Artillery"), "Artillery")
        # Capital Soldier (KIIT) and lower-case soldier (MV) are different classes.
        self.assertEqual(cnn_label("Soldier"), "Soldier")
        self.assertEqual(cnn_label("soldier"), "camouflage_soldier")
        # All vehicle synonyms collapse to the same CNN label.
        self.assertEqual(cnn_label("Tank"), "military_vehicle")
        self.assertEqual(cnn_label("tank"), "military_vehicle")
        self.assertEqual(cnn_label("armoured personnel carrier"), "military_vehicle")
        self.assertEqual(cnn_label("military_vehicle"), "military_vehicle")

    def test_dropped_yolo_names_return_none(self) -> None:
        # These classes are not in the CNN set and must be silently dropped.
        self.assertIsNone(cnn_label("Vehicle"))
        self.assertIsNone(cnn_label("weapon"))
        self.assertIsNone(cnn_label("civilian"))
        self.assertIsNone(cnn_label("civilian_vehicle"))
        self.assertIsNone(cnn_label("unknown_class"))

    def test_shared_list_has_exactly_ten_labels(self) -> None:
        self.assertEqual(len(CNN_LABELS), 10)

    def test_every_yolo_mapping_target_is_in_the_shared_list(self) -> None:
        # No YOLO mapping can point to a name outside CNN_LABELS.
        for raw, mapped in YOLO_TO_CNN.items():
            self.assertIn(mapped, CNN_LABELS, msg=f"YOLO_TO_CNN['{raw}'] = '{mapped}' is not in CNN_LABELS")

    def test_three_weight_files_are_listed(self) -> None:
        self.assertEqual(len(YOLO_FILES), 3)
        for name in YOLO_FILES:
            self.assertTrue(name.endswith(".pt"), msg=f"Expected .pt extension: {name}")

    def test_path_constants_use_expected_names(self) -> None:
        self.assertEqual(VIDEO_PATH.name, "combined_drone.mp4")
        self.assertEqual(EFFNET_PATH.name, "EffNet_b0.pt")


if __name__ == "__main__":
    unittest.main()
