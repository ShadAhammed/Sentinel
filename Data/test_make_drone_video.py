"""Tests for the synthetic drone flight settings."""

import unittest

import numpy as np

import make_drone_video


class DroneVideoTests(unittest.TestCase):
    def test_video_is_thirty_seconds_at_32_fps(self) -> None:
        self.assertEqual(make_drone_video.FPS, 32)
        self.assertEqual(make_drone_video.SECONDS, 30)
        self.assertEqual(make_drone_video.FRAME_COUNT, 960)
        self.assertEqual(make_drone_video.SCENE_FRAMES * len(make_drone_video.LABELS), 960)

    def test_every_label_gets_its_own_scene(self) -> None:
        self.assertEqual(len(make_drone_video.LABELS), 10)
        for frame_index in (0, 95, 96, 959):
            scene, _local = make_drone_video.frame_scene(frame_index)
            self.assertGreaterEqual(scene, 0)
            self.assertLess(scene, 10)

    def test_ken_burns_returns_a_720p_view(self) -> None:
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        view = make_drone_video.ken_burns(image, 0, make_drone_video.SCENE_FRAMES)
        self.assertEqual(view.shape, (make_drone_video.VIEW_H, make_drone_video.VIEW_W, 3))


if __name__ == "__main__":
    unittest.main()
