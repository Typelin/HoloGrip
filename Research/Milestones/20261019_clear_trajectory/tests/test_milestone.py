"""Offline artifact/contract tests; no private performance dataset is bundled."""
import ast
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Program"))
import world_stream_runtime_1019 as runtime


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def signature(rows):
    return sorted((r["hand"], float(r["peak_ms"]), float(r["emitted_ms"]),
                   r["drum"], round(float(r["model_score"]), 6)) for r in rows)


def synthetic_rows():
    """Artificial two-hand impulses, not a recording or classification truth."""
    for i in range(180):
        for hand in "LR":
            distance = min(abs(i - 60), abs(i - 120))
            impulse = max(0.0, 1.0 - distance / 3.0) * 4.0
            yield dict(hand=hand, song_time_ms=i * 10.0,
                       sensor_time_ms=10000.0 + i * 10.0,
                       ax_g=impulse, ay_g=0.05, az_g=1.0,
                       yaw_deg=10.0 if hand == "L" else -15.0,
                       pitch_deg=0.0, roll_deg=0.0,
                       cal_yaw_deg=0.0, cal_pitch_deg=0.0)


class MilestoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_json(ROOT / "manifest.json")
        cls.metrics = read_json(ROOT / "evidence/metrics.json")
        cls.evidence = read_json(ROOT / "evidence/verification.json")
        cls.bundle = joblib.load(ROOT / "Model/model.joblib")

    def test_frozen_bytes_and_syntax(self):
        for name, expected in self.manifest["files"].items():
            path = ROOT / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, name)
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=name)
        self.assertEqual(len(self.manifest["files"]), 9)

    def test_model_architecture_and_fit_membership(self):
        bundle, manifest = self.bundle, self.manifest
        self.assertEqual(bundle["runtime_contract"], runtime.world.CONTRACT)
        self.assertEqual(bundle["kind"], "world_motion")
        self.assertEqual(set(bundle["models"]), {"L", "R"})
        self.assertEqual(len(bundle["training_keys"]), 6667)
        self.assertEqual(len(set(bundle["training_keys"])), 6667)
        counts = collections.Counter(key.split(":")[0] for key in bundle["training_keys"])
        self.assertEqual(dict(counts), manifest["training_recording_counts"])
        self.assertTrue(set(counts).isdisjoint({"round4", "round5", "round6"}))
        self.assertFalse(bundle["pretest_training_overlap"])
        self.assertFalse(bundle["additional_calibration"])
        self.assertFalse(bundle["answer_derived_anchor"])
        self.assertFalse(bundle["abstention"])
        self.assertEqual(bundle["detector_config"], manifest["detector_config"])
        self.assertEqual(len(bundle["spec"]["columns"]), 219)
        rows_seen = 0
        for hand, model in bundle["models"].items():
            self.assertEqual(model.n_features_in_, 417)
            mlp = model.named_steps["model"]
            self.assertEqual(mlp.hidden_layer_sizes, (256, 128))
            self.assertEqual(mlp.n_features_in_, 219)
            self.assertEqual(mlp.alpha, 0.3)
            self.assertFalse(mlp.early_stopping)
            seen = int(model.named_steps["scaler"].n_samples_seen_)
            self.assertEqual(seen, manifest["scaler_rows"][hand])
            rows_seen += seen
        self.assertEqual(rows_seen, 6667)

    def test_score_arithmetic_and_evaluation_roles(self):
        data = self.metrics
        for row in data["full_recordings"] + data["normal_velocity_recordings"]:
            self.assertEqual(row["correct"] + row["wrong_drum"], row["time_hand_matched"])
            self.assertEqual(row["time_hand_matched"] + row["missed"], row["total"])
        for key in ["total", "correct", "time_hand_matched", "missed", "wrong_drum", "unmatched_extra"]:
            self.assertEqual(sum(row[key] for row in data["full_recordings"]), data["full_aggregate"][key])
        self.assertEqual(data["full_aggregate"]["correct"], 6636)
        self.assertEqual(data["full_aggregate"]["total"], 7079)
        for row in data["pretests"]:
            self.assertEqual(row["correct"] + row["wrong_drum"], row["matched"])
            self.assertEqual(row["matched"] + row["missed"], row["total"])
            self.assertGreaterEqual(row["correct"] / row["total"], 0.87)
        aggregate = data["pretest_aggregate"]
        self.assertEqual((aggregate["correct"], aggregate["total"]), (529, 594))
        self.assertEqual(sum(row["correct"] for row in data["pretests"]), aggregate["correct"])
        self.assertEqual(sum(row["total"] for row in data["pretests"]), aggregate["total"])
        self.assertFalse(data["unseen_round_holdout"])
        self.assertIn("reused", data["pretest_role"])
        self.assertEqual(data["normal_velocity_policy"]["midi_velocity_min"], 20)
        self.assertFalse(data["normal_velocity_policy"]["raw_peak_or_model_success_filter"])
        self.assertFalse(data["extra_outputs_penalize_hit_rate"])

    def test_private_validation_summary_consistency(self):
        evidence = self.evidence
        self.assertTrue(evidence["no_hardware_test"])
        self.assertTrue(evidence["pretests_not_in_fit_or_scaler"])
        self.assertEqual(evidence["independent_raw_only_cases"], 13)
        self.assertEqual(evidence["prefix_feature_cases"], 39)
        self.assertEqual(len(evidence["raw_only_replay_manifests"]), 13)
        pretests = [evidence["raw_only_replay_manifests"][name] for name in ["round4", "round5", "round6"]]
        self.assertEqual(sum(row["raw_rows"] for row in pretests), 169120)
        self.assertEqual(sum(row["predictions"] for row in pretests), 850)
        for row in evidence["raw_only_replay_manifests"].values():
            self.assertEqual(row["model_sha256"], evidence["model_sha256"])
            self.assertFalse(row["midi_or_labels_read_during_inference"])
            self.assertFalse(row["additional_calibration"])

    def test_synthetic_packet_batch_parity_and_causal_prefixes(self):
        with tempfile.TemporaryDirectory(prefix="hologrip-milestone-") as tmp:
            raw = Path(tmp) / "synthetic.csv"
            rows = list(synthetic_rows())
            with raw.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            hands = runtime.world.read_raw(raw)
            peaks, _ = runtime.world.base.detector.detect(hands, self.bundle["detector_config"])
            with threadpool_limits(limits=1):
                batch = runtime.predict(self.bundle, hands, peaks)
                engine = runtime.StreamEngine(self.bundle)
                streamed = []
                for _, hand, row in sorted((r["row_index"], h, r) for h, values in hands.items() for r in values):
                    streamed.extend(engine.push(hand, row))
            self.assertEqual(len(batch), 4)
            self.assertEqual(signature(batch), signature(streamed))
            prepared = runtime.world.base.legacy.prepare(hands)
            for peak in peaks:
                prefix = {h: values[:int(peak["available_index"]) + 1] if h == peak["hand"] else []
                          for h, values in hands.items()}
                expected = runtime.world.feature(prepared, peak, "world_motion")
                actual = runtime.world.feature(runtime.world.base.legacy.prepare(prefix), peak, "world_motion")
                self.assertEqual(expected.shape, (417,))
                np.testing.assert_array_equal(expected, actual)
            cli = [sys.executable, str(ROOT / "Program/world_stream_runtime_1019.py"),
                   "--raw", str(raw), "--model", str(ROOT / "Model/model.joblib")]
            output = Path(tmp) / "stream.csv"
            result = subprocess.run(cli + ["--output", str(output), "--stream"],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            meta = read_json(output.with_suffix(".manifest.json"))
            self.assertEqual(meta["predictions"], 4)
            self.assertTrue(meta["actual_packet_stream"])
            original = output.read_bytes()
            refused = subprocess.run(cli + ["--output", str(output), "--stream"],
                                     capture_output=True, text=True, timeout=60)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
