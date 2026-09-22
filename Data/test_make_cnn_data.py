"""Tests for the CNN-Data label drops, merges, and caps."""

import random
import unittest

import make_cnn_data


def row(split: str, label: str, filename: str) -> dict[str, str]:
    """Build one labels.csv row with a stable crop path."""
    return {
        "split": split,
        "label": label,
        "dataset": "test",
        "source_image": "source.jpg",
        "crop_file": f"{split}/{label}/{filename}",
    }


class MakeCnnDataTests(unittest.TestCase):
    def test_drop_and_merge_labels(self) -> None:
        self.assertIsNone(make_cnn_data.mapped_label("Vehicle"))
        self.assertIsNone(make_cnn_data.mapped_label("pedestrian"))
        self.assertIsNone(make_cnn_data.mapped_label("awning-tricycle"))
        self.assertEqual(make_cnn_data.mapped_label("military_vehicle"), "military_vehicle")
        self.assertEqual(make_cnn_data.mapped_label("military_truck"), "military_vehicle")
        self.assertEqual(make_cnn_data.mapped_label("Tank"), "military_vehicle")
        self.assertEqual(make_cnn_data.mapped_label("camouflage_soldier"), "camouflage_soldier")
        self.assertIsNone(make_cnn_data.mapped_label("civilian_vehicle"))
        self.assertIsNone(make_cnn_data.mapped_label("civilian"))
        self.assertIsNone(make_cnn_data.mapped_label("weapon"))
        self.assertEqual(make_cnn_data.mapped_label("military_tank"), "military_vehicle")
        self.assertEqual(make_cnn_data.mapped_label("military_artillery"), "Artillery")
        self.assertEqual(make_cnn_data.mapped_label("Artilary"), "Artillery")
        self.assertEqual(make_cnn_data.mapped_label("Soldier_KIIT-MiTA"), "Soldier")
        self.assertEqual(make_cnn_data.mapped_label("soldier_military_object_dataset"), "Soldier")
        self.assertEqual(make_cnn_data.mapped_label("soldier"), "Soldier")

    def test_training_cap_keeps_validation_then_fills_train(self) -> None:
        rows = []
        for index in range(3):
            rows.append(row("validation", "Radar", f"val_{index}.jpg"))
        for index in range(1200):
            rows.append(row("train", "Radar", f"train_{index}.jpg"))
        for index in range(80):
            rows.append(row("test", "Radar", f"test_{index}.jpg"))

        chosen = make_cnn_data.select_rows(rows, random.Random(42))
        counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        for item in chosen:
            counts[item["split"]] += 1
            self.assertEqual(item["label"], "Radar")

        self.assertEqual(counts["validation"], 3)
        self.assertEqual(counts["train"], 997)
        self.assertEqual(counts["test"], 50)

    def test_fresh_copy_caps_validation_before_train(self) -> None:
        rows = []
        for index in range(150):
            rows.append(row("validation", "military_aircraft", f"val_{index}.jpg"))
        for index in range(2000):
            rows.append(row("train", "military_aircraft", f"train_{index}.jpg"))

        chosen = make_cnn_data.select_rows(rows, random.Random(42), validation_max=100)
        counts: dict[str, int] = {"train": 0, "validation": 0}
        for item in chosen:
            counts[item["split"]] += 1

        self.assertEqual(counts["validation"], 100)
        self.assertEqual(counts["train"], 900)

    def test_small_label_is_kept_and_path_is_renamed(self) -> None:
        rows = [
            row("train", "military_tank", "tank_a.jpg"),
            row("validation", "Artilary", "gun_a.jpg"),
            row("test", "military_artillery", "gun_b.jpg"),
            row("train", "car", "car_a.jpg"),
        ]
        chosen = make_cnn_data.select_rows(rows, random.Random(42))
        by_file = {item["source_crop"]: item for item in chosen}

        self.assertNotIn("train/car/car_a.jpg", by_file)
        self.assertEqual(
            by_file["train/military_tank/tank_a.jpg"]["crop_file"],
            "train\\military_vehicle\\tank_a.jpg",
        )
        self.assertEqual(by_file["validation/Artilary/gun_a.jpg"]["label"], "Artillery")
        self.assertEqual(by_file["test/military_artillery/gun_b.jpg"]["crop_file"], "test\\Artillery\\gun_b.jpg")


if __name__ == "__main__":
    unittest.main()
