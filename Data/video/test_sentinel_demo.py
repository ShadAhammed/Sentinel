"""Tests for the SENTINEL-X demo window helpers."""

import unittest

import numpy as np

import sentinel_demo


class SentinelDemoTests(unittest.TestCase):
    def test_credit_lists_the_three_datasets(self) -> None:
        credits = sentinel_demo.dataset_credits()
        titles = [item["title"] for item in credits]
        self.assertEqual(titles, ["KIIT-MiTA", "military_object_dataset", "MV"])
        joined = "\n".join(item["body"] for item in credits)
        self.assertIn("10.17632/drjmrf5kk5.1", joined)
        self.assertIn("CC BY 4.0", joined)
        self.assertIn("Ryan Madhuwala", joined)

    def test_fit_frame_keeps_the_picture_inside_the_panel(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        fitted = sentinel_demo.fit_frame(frame, 820, 500)
        height, width = fitted.shape[:2]
        self.assertLessEqual(width, 820)
        self.assertLessEqual(height, 500)
        self.assertAlmostEqual(width / height, 1280 / 720, places=2)

    def test_cnn_name_wins_when_the_models_differ(self) -> None:
        self.assertEqual(sentinel_demo.final_label("Soldier", "camouflage_soldier"), "camouflage_soldier")
        self.assertEqual(sentinel_demo.final_label("Soldier", "Soldier"), "Soldier")
        self.assertEqual(sentinel_demo.final_label("Soldier", None), "Soldier")
        kept = sentinel_demo.efficientnet_hits(
            [
                {"xyxy": [0, 0, 40, 40], "cnn_name": None, "cnn_confidence": None},
                {"xyxy": [0, 0, 100, 100], "cnn_name": "Soldier", "cnn_confidence": 0.4},
                {"xyxy": [10, 10, 110, 110], "cnn_name": "military_vehicle", "cnn_confidence": 0.9},
                {"xyxy": [400, 400, 500, 500], "cnn_name": "Soldier", "cnn_confidence": 0.7},
            ]
        )
        names = sorted(item["cnn_name"] for item in kept)
        self.assertEqual(names, ["Soldier", "military_vehicle"])

    def test_clock_and_cnn_caption(self) -> None:
        self.assertEqual(sentinel_demo.format_clock(0), "00:00")
        self.assertEqual(sentinel_demo.format_clock(75.8), "01:15")
        self.assertEqual(sentinel_demo.parse_clock("01:15"), 75)
        self.assertEqual(sentinel_demo.cnn_caption("Soldier", 0.88), "Soldier 88%")
        self.assertIsNone(sentinel_demo.cnn_caption(None, None))
        self.assertIn("Sentinel, version 1.0", sentinel_demo.GREETING)
        self.assertIn("potential dangers", sentinel_demo.GREETING)
        self.assertEqual(len(sentinel_demo.GREETING.split()), 60)
        self.assertEqual(sentinel_demo.split_view(1012), (400, 600))
        self.assertEqual(
            sentinel_demo.ssh_client_display("192.168.55.100 50000 192.168.55.1 22"),
            "192.168.55.100:0.0",
        )
        self.assertIsNone(sentinel_demo.ssh_client_display(""))
        # A forwarded display (ssh -Y from a Mac) must be kept, not overwritten.
        self.assertTrue(sentinel_demo.is_forwarded_display("localhost:10.0"))
        self.assertTrue(sentinel_demo.is_forwarded_display("127.0.0.1:10.0"))
        self.assertFalse(sentinel_demo.is_forwarded_display("192.168.55.100:0.0"))
        data_dir, video = sentinel_demo.demo_dirs()
        self.assertEqual(video.name, "combined_drone.mp4")
        self.assertTrue((data_dir / "jetson_test.py").is_file())

    def test_video_context_lists_categories(self) -> None:
        empty = sentinel_demo.format_detection_context(
            {
                "frames_seen": 0,
                "latest": [],
                "clock": "00:00",
                "totals": {},
                "frames": {},
                "maximums": {},
            }
        )
        self.assertIn("no object counts", empty)
        running = {
            "finished": False,
            "clock": "00:12",
            "playhead": 12.0,
            "frame_log": [
                {"seconds": 10.0, "clock": "00:10", "hits": [("Soldier", 0.9), ("Soldier", 0.7)]},
                {"seconds": 20.0, "clock": "00:20", "hits": [("military_warship", 0.8)]},
            ],
            "crops": [],
        }
        text = sentinel_demo.format_detection_context(running)
        self.assertIn("still running", text)
        self.assertIn("Soldier: at most 2 at once", text)
        self.assertNotIn("military_warship", text)
        running["finished"] = True
        running["crops"] = [
            ("Soldier", 0.9, "00:10"),
            ("military_warship", 0.8, "00:20"),
        ]
        # A finished reader must not hand the model frames after the picture.
        early = sentinel_demo.format_detection_context(running)
        self.assertNotIn("military_warship", early)
        self.assertIn("Soldier at 00:10", early)
        running["playhead"] = 20.0
        running["clock"] = "00:20"
        finished = sentinel_demo.format_detection_context(running)
        self.assertIn("Playback is complete", finished)
        self.assertIn("military_warship", finished)
        self.assertEqual(sentinel_demo.asked_object("show me the warship"), "military_warship")
        self.assertEqual(sentinel_demo.asked_object("show me the tank"), "military_vehicle")
        self.assertIsNone(sentinel_demo.asked_object("does this look like a war zone"))
        self.assertIsNone(sentinel_demo.asked_object("how many soldiers"))

    def test_confidence_is_a_percent(self) -> None:
        self.assertEqual(sentinel_demo.confidence_percent(0.876), "88%")
        self.assertEqual(sentinel_demo.parse_gpu_line("35, 2048, 8192"), (35, 2048, 8192))
        self.assertEqual(sentinel_demo.format_gpu_free(8192), "8.0 GB")

    def test_chat_takes_forty_percent_of_the_row(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        window = sentinel_demo.DemoWindow(root)
        window._on_body_resize(type("Event", (), {"widget": window.body, "width": 1012})())
        self.assertEqual(int(window.side.cget("width")), 400)
        window.close()

    def test_window_has_a_credit_tab(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        window = sentinel_demo.DemoWindow(root)
        self.assertEqual(window.tab_names(), ["Detection", "Credit"])
        self.assertEqual(window.play_button.cget("text"), "Play")
        self.assertNotIn("version 1.0", window.chat.get("1.0", "end"))
        self.assertEqual(window.send_button.cget("state"), "disabled")
        window._begin_intro()
        self.assertIn("Sentinel", window.chat.get("1.0", "end"))
        self.assertNotIn("version", window.chat.get("1.0", "end"))
        window._type_intro()
        self.assertTrue(window.chat.get("1.0", "end").rstrip().endswith("I"))
        self.assertEqual(window.chat.cget("background"), sentinel_demo.CHAT_BG)
        self.assertEqual(window.chat.cget("foreground"), sentinel_demo.MATRIX)
        self.assertEqual(window.time_now.cget("text"), "00:00")
        window.report["frame_log"] = [
            {"seconds": 2.0, "clock": "00:02", "hits": [("Soldier", 0.5)]},
            {"seconds": 20.0, "clock": "00:20", "hits": [("military_warship", 0.8)]},
        ]
        window.crops = {
            "Soldier": [(0.5, None, "00:02", 2.0)],
            "military_warship": [(0.8, None, "00:20", 20.0)],
        }
        window.shown_seconds = 20.0
        window.ended = True
        with window.lock:
            window._cut_log(2.0)
        context = window._video_context()
        self.assertNotIn("military_warship", context)
        self.assertIn("Soldier", context)
        self.assertIn("still running", context)
        window.close()


if __name__ == "__main__":
    unittest.main()
