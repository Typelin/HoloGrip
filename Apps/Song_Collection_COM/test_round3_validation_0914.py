import csv
import tempfile
import time
import unittest
from pathlib import Path

import analyze_round3_validation_0914_ZH_TW as ana
import run_round3_validation_0914_ZH_TW as r3
import round3_capture_0914_ZH_TW as capture


class Round3ValidationChecks(unittest.TestCase):
    def stable_samples(self, yaw_step=0.0, pitch_step=0.0):
        out=[]
        for i in range(500):
            out.append({
                "mono": i*0.01,
                "packet_id": i,
                "mag": 1.0,
                "cal_yaw": yaw_step*i/499.0,
                "cal_pitch": pitch_step*i/499.0,
                "zero": False,
            })
        return out

    def test_preflight_pass_and_drift_warn(self):
        ok=r3.compute_preflight(self.stable_samples(1.0,1.0))
        self.assertEqual(ok["status"],"PASS")
        self.assertGreater(ok["hz"],99.0)
        bad=r3.compute_preflight(self.stable_samples(12.0,9.0))
        self.assertEqual(bad["status"],"WARN")
        self.assertTrue(any("Yaw" in x for x in bad["reasons"]))

    def test_recorder_writes_all_evidence_files(self):
        with tempfile.TemporaryDirectory() as td:
            old=capture.VALIDATION_ROOT
            try:
                capture.VALIDATION_ROOT=Path(td)
                rec=r3.ValidationRecorder()
                session=rec.start("R3_TEST",Path(__file__),"abc")
                rec.raw({"session_id":"R3_TEST","session_time_ms":1,"host_time_ms":1,"phase":"setup","port":"COM1","hand":"L","packet_id":1,"sensor_time_ms":1,"calibrated":0,"ax_g":0,"ay_g":0,"az_g":1,"accel_magnitude_g":1,"yaw_deg":0,"pitch_deg":0,"roll_deg":0,"cal_yaw_deg":0,"cal_pitch_deg":0,"zero_packet":0})
                rec.pred({"session_id":"R3_TEST","phase":"song","peak_session_ms":100,"hand":"L","pred_drum":"小鼓","pred_proba":0.9,"peak_accel_g":3,"model_sha256":"abc"})
                rec.marker("SONG_START","test")
                rec.stop()
                self.assertTrue((session/"raw_100hz.csv").exists())
                self.assertTrue((session/"predictions.csv").exists())
                self.assertTrue((session/"markers.csv").exists())
                with (session/"predictions.csv").open(encoding="utf-8-sig") as f:
                    rows=list(csv.DictReader(f))
                self.assertEqual(rows[0]["pred_drum"],"小鼓")
            finally:
                capture.VALIDATION_ROOT=old

    def test_slow_marker_diagnostic(self):
        markers=[
            {"marker":"SLOW_DRUM","session_time_ms":"1000","note":"小鼓"},
            {"marker":"SLOW_DRUM","session_time_ms":"2000","note":"Hi-Hat"},
        ]
        preds=[
            {"phase":"slow_test","peak_session_ms":"1100","pred_drum":"小鼓"},
            {"phase":"slow_test","peak_session_ms":"1200","pred_drum":"高音 Tom"},
            {"phase":"slow_test","peak_session_ms":"2100","pred_drum":"Hi-Hat"},
        ]
        rep=ana.slow_test_diagnostic(preds,markers)
        self.assertEqual(rep["detected_predictions"],3)
        self.assertEqual(rep["correct_on_detected"],2)

    def test_temporal_matching_and_exact(self):
        preds=[{"time_ms":1000,"drum":"小鼓"},{"time_ms":1500,"drum":"Hi-Hat"}]
        midi=[{"time_ms":0,"drum":"小鼓"},{"time_ms":500,"drum":"Hi-Hat"}]
        pairs=ana.greedy_temporal_pairs(preds,midi,1000,50)
        self.assertEqual(len(pairs),2)
        self.assertEqual(ana.greedy_exact_count(preds,midi,1000,50),2)

    def test_frozen_model_hash(self):
        model=Path(r3.live._resolve_model_path())
        self.assertTrue(model.is_file())
        self.assertEqual(r3.sha256_file(model).lower(),r3.EXPECTED_MODEL_SHA256)

    def test_gui_constructs_with_validation_controls(self):
        from unittest.mock import patch
        with patch.object(r3.live.LiveHitZoneApp,"_refresh_ports"):
            app=r3.Round3ValidationApp()
        app.update(); app.update_idletasks()
        try:
            self.assertTrue(app.start_session_btn.winfo_exists())
            self.assertTrue(app.stop_session_btn.winfo_exists())
            self.assertFalse(app.recorder.active)
            self.assertFalse(app.session_calibrated)
        finally:
            app._close()


if __name__ == "__main__":
    unittest.main()
