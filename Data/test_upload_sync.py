"""Tests for the Edge Impulse sync decision."""

import unittest

from upload_to_edge_impulse import plan_remote_sample


class UploadSyncTests(unittest.TestCase):
    def test_keeps_matching_sample_and_deletes_duplicates(self) -> None:
        desired = {("training", "tank_a"): "Tank"}
        seen: set[tuple[str, str]] = set()

        self.assertEqual(
            plan_remote_sample("tank_a.jpg", "Tank", "training", desired, seen),
            ("keep", "Tank"),
        )
        self.assertEqual(
            plan_remote_sample("tank_a", "Tank", "training", desired, seen),
            ("delete", None),
        )

    def test_relabels_merged_sample_and_deletes_removed_label(self) -> None:
        desired = {("training", "gun_a"): "Artillery"}
        seen: set[tuple[str, str]] = set()

        self.assertEqual(
            plan_remote_sample("gun_a", "Artilary", "training", desired, seen),
            ("relabel", "Artillery"),
        )
        self.assertEqual(
            plan_remote_sample("car_a", "car", "training", desired, seen),
            ("delete", None),
        )


if __name__ == "__main__":
    unittest.main()
