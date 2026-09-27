"""Tests for the demo window helpers and the DemoWindow class itself."""

import unittest

import numpy as np

from src.chat import GREETING, asked_object
from src.gpu import format_gpu_free, parse_gpu_line
from src.window import (
    CHAT_BG,
    MATRIX,
    DemoWindow,
    cnn_caption,
    dataset_credits,
    efficientnet_hits,
    fit_frame,
    format_clock,
    format_detection_context,
    is_forwarded_display,
    parse_clock,
    split_view,
    ssh_client_display,
)


class ClockTests(unittest.TestCase):

    def test_format_clock_converts_seconds_to_mm_ss(self) -> None:
        self.assertEqual(format_clock(0), "00:00")
        self.assertEqual(format_clock(75.8), "01:15")
        self.assertEqual(format_clock(3599), "59:59")

    def test_parse_clock_converts_mm_ss_to_seconds(self) -> None:
        self.assertEqual(parse_clock("00:00"), 0)
        self.assertEqual(parse_clock("01:15"), 75)


class CaptionTests(unittest.TestCase):

    def test_cnn_caption_formats_label_and_percent(self) -> None:
        self.assertEqual(cnn_caption("Soldier", 0.88), "Soldier 88%")

    def test_cnn_caption_returns_none_when_crop_was_skipped(self) -> None:
        self.assertIsNone(cnn_caption(None, None))
        self.assertIsNone(cnn_caption(None, 0.9))


class SplitViewTests(unittest.TestCase):

    def test_chat_takes_forty_percent(self) -> None:
        chat, video = split_view(1012)
        self.assertEqual(chat, 400)
        self.assertEqual(video, 600)


class FitFrameTests(unittest.TestCase):

    def test_frame_is_shrunk_to_fit_panel(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        fitted = fit_frame(frame, 820, 500)
        h, w = fitted.shape[:2]
        self.assertLessEqual(w, 820)
        self.assertLessEqual(h, 500)
        self.assertAlmostEqual(w / h, 1280 / 720, places=2)

    def test_small_frame_is_not_upscaled(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        fitted = fit_frame(frame, 800, 600)
        self.assertEqual(fitted.shape[:2], (100, 200))


class CreditsTests(unittest.TestCase):

    def test_three_datasets_are_listed(self) -> None:
        credits = dataset_credits()
        titles = [c["title"] for c in credits]
        self.assertEqual(titles, ["KIIT-MiTA", "military_object_dataset", "MV"])

    def test_doi_and_license_are_present(self) -> None:
        joined = "\n".join(c["body"] for c in dataset_credits())
        self.assertIn("10.17632/drjmrf5kk5.1", joined)
        self.assertIn("CC BY 4.0", joined)
        self.assertIn("Ryan Madhuwala", joined)


class GpuTests(unittest.TestCase):

    def test_parse_gpu_line_reads_three_csv_integers(self) -> None:
        self.assertEqual(parse_gpu_line("35, 2048, 8192"), (35, 2048, 8192))

    def test_parse_gpu_line_returns_none_for_bad_data(self) -> None:
        self.assertIsNone(parse_gpu_line("bad"))
        self.assertIsNone(parse_gpu_line("1, 2"))

    def test_format_gpu_free_uses_gigabytes(self) -> None:
        self.assertEqual(format_gpu_free(8192), "8.0 GB")
        self.assertEqual(format_gpu_free(1024), "1.0 GB")


class GreetingTests(unittest.TestCase):

    def test_greeting_identifies_sentinel_and_contains_sixty_words(self) -> None:
        self.assertIn("Sentinel, version 1.0", GREETING)
        self.assertIn("potential dangers", GREETING)
        self.assertEqual(len(GREETING.split()), 60)


class SshDisplayTests(unittest.TestCase):

    def test_ssh_connection_string_gives_client_display(self) -> None:
        result = ssh_client_display("192.168.55.100 50000 192.168.55.1 22")
        self.assertEqual(result, "192.168.55.100:0.0")

    def test_empty_connection_string_returns_none(self) -> None:
        self.assertIsNone(ssh_client_display(""))

    def test_forwarded_display_is_recognized(self) -> None:
        self.assertTrue(is_forwarded_display("localhost:10.0"))
        self.assertTrue(is_forwarded_display("127.0.0.1:10.0"))
        self.assertFalse(is_forwarded_display("192.168.55.100:0.0"))


class AskedObjectTests(unittest.TestCase):

    def test_show_warship_maps_to_military_warship(self) -> None:
        self.assertEqual(asked_object("show me the warship"), "military_warship")

    def test_show_tank_maps_to_military_vehicle(self) -> None:
        self.assertEqual(asked_object("show me the tank"), "military_vehicle")

    def test_question_without_show_word_returns_none(self) -> None:
        self.assertIsNone(asked_object("does this look like a war zone"))
        self.assertIsNone(asked_object("how many soldiers"))


class DetectionContextTests(unittest.TestCase):

    def test_empty_log_reports_no_counts(self) -> None:
        text = format_detection_context({"frame_log": [], "playhead": 0.0})
        self.assertIn("no object counts", text)

    def test_frames_after_playhead_are_excluded(self) -> None:
        report = {
            "finished": False,
            "clock": "00:12",
            "playhead": 12.0,
            "frame_log": [
                {"seconds": 10.0, "clock": "00:10", "hits": [("Soldier", 0.9), ("Soldier", 0.7)]},
                {"seconds": 20.0, "clock": "00:20", "hits": [("military_warship", 0.8)]},
            ],
            "crops": [],
        }
        text = format_detection_context(report)
        self.assertIn("still running", text)
        self.assertIn("Soldier: at most 2 at once", text)
        self.assertNotIn("military_warship", text)

    def test_finished_report_shows_all_detections(self) -> None:
        report = {
            "finished": True,
            "clock": "00:20",
            "playhead": 20.0,
            "frame_log": [
                {"seconds": 10.0, "clock": "00:10", "hits": [("Soldier", 0.9)]},
                {"seconds": 20.0, "clock": "00:20", "hits": [("military_warship", 0.8)]},
            ],
            "crops": [("Soldier", 0.9, "00:10"), ("military_warship", 0.8, "00:20")],
        }
        text = format_detection_context(report)
        self.assertIn("Playback is complete", text)
        self.assertIn("military_warship", text)
        self.assertIn("Soldier at 00:10", text)


class DemoWindowTests(unittest.TestCase):

    def setUp(self) -> None:
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = DemoWindow(self.root)

    def tearDown(self) -> None:
        self.window.close()
        # Wait for the background GPU thread to finish so Tkinter cleanup is clean.
        self.window.gpu_thread.join(timeout=2.0)

    def test_tabs_are_detection_and_credit(self) -> None:
        self.assertEqual(self.window.tab_names(), ["Detection", "Credit"])

    def test_play_button_starts_as_play(self) -> None:
        self.assertEqual(self.window.play_button.cget("text"), "Play")

    def test_send_button_is_disabled_until_greeting_finishes(self) -> None:
        self.assertEqual(self.window.send_button.cget("state"), "disabled")

    def test_intro_starts_after_begin_intro_is_called(self) -> None:
        self.assertNotIn("version 1.0", self.window.chat.get("1.0", "end"))
        self.window._begin_intro()
        self.assertIn("Sentinel", self.window.chat.get("1.0", "end"))
        self.window._type_intro()
        self.assertTrue(self.window.chat.get("1.0", "end").rstrip().endswith("I"))

    def test_chat_panel_uses_matrix_green_on_black(self) -> None:
        self.assertEqual(self.window.chat.cget("background"), CHAT_BG)
        self.assertEqual(self.window.chat.cget("foreground"), MATRIX)

    def test_body_resize_keeps_chat_at_forty_percent(self) -> None:
        self.window._on_body_resize(type("Event", (), {"widget": self.window.body, "width": 1012})())
        self.assertEqual(int(self.window.side.cget("width")), 400)

    def test_cut_log_drops_later_detections(self) -> None:
        self.window.report["frame_log"] = [
            {"seconds": 2.0,  "clock": "00:02", "hits": [("Soldier", 0.5)]},
            {"seconds": 20.0, "clock": "00:20", "hits": [("military_warship", 0.8)]},
        ]
        self.window.crops = {
            "Soldier":           [(0.5, None, "00:02", 2.0)],
            "military_warship":  [(0.8, None, "00:20", 20.0)],
        }
        self.window.shown_seconds = 20.0
        self.window.ended = True
        with self.window.lock:
            self.window._cut_log(2.0)
        context = self.window._video_context()
        self.assertNotIn("military_warship", context)
        self.assertIn("Soldier", context)
        self.assertIn("still running", context)


if __name__ == "__main__":
    unittest.main()
