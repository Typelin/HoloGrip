"""Packet-stream and fixed soft-vote support for frozen world-motion models."""
import argparse
from collections import deque
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import absolute_world_motion_runtime_1019 as world

VOTE_CONTRACT = "hologrip_absolute_world_fixed_vote_1019_v1"


def kinds(bundle):
    return sorted({b["kind"] for b in bundle["components"]}) if bundle["runtime_contract"] == VOTE_CONTRACT else [bundle["kind"]]


def probabilities(bundle, vectors, hands):
    if bundle["runtime_contract"] != VOTE_CONTRACT:
        return world.base.probabilities(bundle, vectors[bundle["kind"]], hands)
    result = None
    for weight, model in zip(bundle["weights"], bundle["components"]):
        p = world.base.probabilities(model, vectors[model["kind"]], hands)
        assert model["classes"] == bundle["classes"]
        result = weight*p if result is None else result+weight*p
    return result


def classify(bundle, peaks, vectors):
    if not peaks:
        return []
    p = probabilities(bundle, vectors, [g["hand"] for g in peaks])
    return [dict(**{k:v for k,v in peak.items() if k != "feature"},
                 drum=bundle["classes"][int(q.argmax())], model_score=float(q.max()),
                 model_margin=float(np.sort(q)[-1]-np.sort(q)[-2])) for peak, q in zip(peaks, p)]


def predict(bundle, hands, peaks):
    prepared = world.base.legacy.prepare(hands)
    names = kinds(bundle)
    vectors, selected = {k:[] for k in names}, []
    for peak in peaks:
        values = {k:world.feature(prepared, peak, k) for k in names}
        if any(v is None for v in values.values()):
            continue
        selected.append(peak)
        for k in names:
            vectors[k].append(values[k])
    return classify(bundle, selected, vectors)


class StreamEngine:
    """Validated Raw sample input, no labels/markers/protocol information.

    Each sample includes original absolute_yaw/absolute_pitch, sensor_t and
    ordinary packet index as well as the legacy packet fields. Quality gating
    and durable Raw storage belong to the field application's input layer.
    """
    def __init__(self, bundle):
        self.bundle = bundle
        self.history = {h:deque(maxlen=1024) for h in "LR"}
        self.hit = {h:world.base.detector.HitStream(h, bundle["detector_config"]) for h in "LR"}
        self.count = 0

    def push(self, hand, sample):
        self.history[hand].append(sample)
        peaks = self.hit[hand].push(sample)
        output = []
        if not peaks:
            return output
        rows = list(self.history[hand])
        prepared = world.base.legacy.prepare({hand:rows})
        first = rows[0]["index"]
        names = kinds(self.bundle)
        for peak in peaks:
            local = {**peak, "peak_index":int(peak["peak_index"])-first,
                     "available_index":len(rows)-1}
            values = {k:world.feature(prepared, local, k) for k in names}
            if any(v is None for v in values.values()):
                continue
            self.count += 1
            stamped = {**peak, "prediction_id":self.count}
            output.extend(classify(self.bundle, [stamped], {k:[v] for k,v in values.items()}))
        return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["raw", "model", "output"]:
        ap.add_argument("--"+key, required=True)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Refuse overwrite")
    bundle = joblib.load(args.model)
    assert bundle["runtime_contract"] in [world.CONTRACT, VOTE_CONTRACT]
    hands = world.read_raw(args.raw)
    with threadpool_limits(limits=1):
        if args.stream:
            engine = StreamEngine(bundle)
            output = []
            for _, hand, row in sorted((r["row_index"], h, r) for h, values in hands.items() for r in values):
                output.extend(engine.push(hand, row))
        else:
            peaks, _ = world.base.detector.detect(hands, bundle["detector_config"])
            output = predict(bundle, hands, peaks)
    target.parent.mkdir(parents=True, exist_ok=True)
    world.base.legacy.rt.write_csv(target, output)
    sha = world.base.legacy.rt.sha256
    meta = dict(raw_sha256=sha(args.raw), model_sha256=sha(args.model),
                runtime_sha256=sha(__file__), feature_runtime_sha256=sha(world.__file__),
                output_sha256=sha(target), predictions=len(output),
                actual_packet_stream=bool(args.stream), raw_rows=sum(map(len,hands.values())),
                midi_or_labels_read_during_inference=False, expected_drum_input=False,
                additional_calibration=False, answer_derived_anchor=False, abstention=False,
                external_detector_changed=False, future_context_ms=80)
    target.with_suffix(".manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
