"""Tests for YOLO label drops and merges."""

import unittest

import localize
import train_yolo26


class YoloLabelTests(unittest.TestCase):
    def test_kiit_keeps_only_cnn_labels(self) -> None:
        names = train_yolo26.cnn_names_for(localize.KIIT_NAMES)
        self.assertEqual(
            names,
            [
                "Artillery",
                "M. Rocket Launcher",
                "Missile",
                "Radar",
                "Soldier",
                "military_vehicle",
            ],
        )
        self.assertNotIn("Vehicle", names)

    def test_military_merges_and_drops(self) -> None:
        names = train_yolo26.cnn_names_for(localize.MILITARY_NAMES)
        self.assertEqual(
            names,
            [
                "Artillery",
                "Soldier",
                "camouflage_soldier",
                "military_aircraft",
                "military_vehicle",
                "military_warship",
                "trench",
            ],
        )
        self.assertNotIn("weapon", names)
        self.assertNotIn("civilian", names)
        self.assertNotIn("civilian_vehicle", names)

    def test_native_keeps_mv_class_ids(self) -> None:
        # Class 2 is tank. Class 4 is outside the 3-class MV list.
        text = "0 0.1 0.1 0.2 0.2\n2 0.5 0.5 0.2 0.2\n4 0.2 0.2 0.2 0.2\n"
        rewritten = train_yolo26.copy_native_text(text, len(train_yolo26.MV_NAMES))
        self.assertEqual(
            rewritten.strip().splitlines(),
            ["0 0.1 0.1 0.2 0.2", "2 0.5 0.5 0.2 0.2"],
        )

    def test_rewrite_drops_vehicle_and_remaps_tank(self) -> None:
        names = train_yolo26.cnn_names_for(localize.KIIT_NAMES)
        class_to_idx = {name: index for index, name in enumerate(names)}
        # Class 5 is Tank. Class 6 is Vehicle and must be removed. Class 0 is Artilary.
        text = "5 0.5 0.5 0.2 0.2\n6 0.1 0.1 0.1 0.1\n0 0.2 0.2 0.3 0.3\n"
        rewritten = train_yolo26.rewrite_yolo_text(text, localize.KIIT_NAMES, class_to_idx)
        lines = rewritten.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], f"{class_to_idx['military_vehicle']} 0.5 0.5 0.2 0.2")
        self.assertEqual(lines[1], f"{class_to_idx['Artillery']} 0.2 0.2 0.3 0.3")


if __name__ == "__main__":
    unittest.main()
