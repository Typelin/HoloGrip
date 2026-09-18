"""Regression checks for field calibration, model loading and per-hand display.

Run with: python -m unittest discover -s Apps/Song_Collection_COM -p test_live_readiness_0905.py
"""
import math
import threading
import unittest
from unittest.mock import patch

import product_hit_and_zone as core
import run_live_hit_and_zone_0821_ZH_TW as gui
from song_collection_server import parse_sensor_line


class FakeSerial:
    def __init__(self, **kwargs):
        self.is_open = True
        self.closed = threading.Event()

    def readline(self):
        self.closed.wait(0.005)
        return b""

    def reset_input_buffer(self):
        pass

    def close(self):
        self.is_open = False
        self.closed.set()


def packet(hand="R", yaw=20, pitch=30, t=10):
    return parse_sensor_line(f"D,{hand},0,0,1,{yaw},{pitch},0,1,1000", 0, t)


class FieldChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = core.load_zone_model(gui._resolve_model_path())

    def test_model_contract(self):
        self.assertEqual(self.model.n_features_in_, 17)
        self.assertEqual(list(self.model.classes_), list(range(7)))
        self.assertNotEqual(core.save_zone_model.__defaults__[0], core.MODEL_PATH)

    def test_absolute_repeated_calibration_and_history_reset(self):
        live = core.LiveRecognizer(self.model, require_calibration=True)
        live.push(packet())
        self.assertEqual(len(live.hands["R"].samples), 0)
        for _ in range(3):
            self.assertEqual(live.calibrate(), ["R"])
            self.assertEqual(live.hands["R"].detector.calibrated(packet()), (0, 0))
            self.assertEqual(len(live.hands["R"].samples), 0)
            self.assertEqual(len(live.hands["R"].detector.buffer), 0)
            self.assertEqual(live.hands["R"].pending_peaks, [])
            live.push(packet())
            live.hands["R"].pending_peaks.append({"peak_ms": 10000, "peak_accel_g": 5})
        live.push(packet(yaw=70, pitch=-10))
        live.calibrate()
        self.assertEqual(live.hands["R"].detector.calibrated(packet(yaw=70, pitch=-10)), (0, 0))

    def test_per_hand_gate_and_invalid_values(self):
        live = core.LiveRecognizer(self.model, require_calibration=True)
        live.push(packet("L"))
        live.calibrate()
        live.push(packet("R"))
        self.assertEqual(len(live.hands["R"].samples), 0)
        live.push(packet("L", yaw=math.nan))
        self.assertEqual(len(live.hands["L"].samples), 0)

    def test_gui_retains_both_hits_and_rejects_stale_generation(self):
        with patch.object(gui.LiveHitZoneApp, "_refresh_ports"):
            app = gui.LiveHitZoneApp()
        app.update()
        app.update_idletasks()
        try:
            app.connected = True
            for hand, drum in (("L", "小鼓"), ("R", "Hi-Hat")):
                app.ui_queue.put(("hit", {"hand": hand, "pred_drum": drum,
                    "pred_proba": .9, "peak_accel_g": 4, "generation": 0}))
            app._drain()
            self.assertEqual(app.hand_hit_vars["L"].get(), "小鼓")
            self.assertEqual(app.hand_hit_vars["R"].get(), "Hi-Hat")
            app.recognizer.generation += 1
            app.ui_queue.put(("hit", {"hand": "L", "pred_drum": "Crash", "generation": 0}))
            app._drain()
            self.assertEqual(app.hand_hit_vars["L"].get(), "小鼓")
            app.update_idletasks()
            self.assertGreater(app.cal_btn.winfo_width(), 30)
            self.assertLessEqual(app.cal_btn.winfo_rootx() + app.cal_btn.winfo_width(),
                                 app.winfo_rootx() + app.winfo_width())
            app.com_a_var.set("COM4")
            app.com_b_var.set("COM5")
            with patch("serial.Serial", FakeSerial), patch.object(gui.time, "sleep"):
                app._connect()
                first = app.recognizer
                old_threads = list(app.serial_threads)
                first.push(packet())
                first.calibrate()
                app.ui_queue.put(("hit", {"generation": -1}))
                app._connect()
                self.assertIsNot(app.recognizer, first)
                self.assertTrue(app.recognizer.require_calibration)
                self.assertEqual(app.recognizer.calibrated_hands, set())
                self.assertTrue(app.ui_queue.empty())
                self.assertTrue(all(not t.is_alive() for t in old_threads))
                app._disconnect()
                self.assertFalse(app.connected)
                self.assertFalse(app.serial_threads)
        finally:
            app._close()


if __name__ == "__main__":
    unittest.main()
