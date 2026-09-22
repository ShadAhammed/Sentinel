"""Tests for the EfficientNet crop-size gate."""

import unittest

import EffNet


class EffNetTests(unittest.TestCase):
    def test_keep_crop_refuses_short_side_under_32(self) -> None:
        self.assertFalse(EffNet.keep_crop(31, 200))
        self.assertFalse(EffNet.keep_crop(200, 31))
        self.assertFalse(EffNet.keep_crop(10, 10))
        self.assertFalse(EffNet.keep_crop(32, 10))
        self.assertTrue(EffNet.keep_crop(32, 32))
        self.assertTrue(EffNet.keep_crop(49, 93))
        self.assertEqual(EffNet.IMAGE_SIZE, 224)
        self.assertEqual(EffNet.MIN_SHORT_SIDE, 32)


if __name__ == "__main__":
    unittest.main()
