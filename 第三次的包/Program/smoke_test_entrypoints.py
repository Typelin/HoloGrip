from __future__ import annotations

import sys


def collector() -> None:
    import song_collection_com  # noqa: F401
    import song_collection_server

    assert "product_hit_and_zone" not in sys.modules
    assert song_collection_server.COLLECTION_OUTPUT_DIRS["serial"].parts[-2:] == ("Data", "CSV")
    print("COLLECTOR_ENTRY_OK")


def model() -> None:
    import run_live_hit_and_zone_0821_ZH_TW as live

    clf = live.load_zone_model(live._resolve_model_path())
    assert clf.n_features_in_ == 17
    assert len(clf.classes_) == 7
    print("MODEL_ENTRY_OK")



def formal() -> None:
    import formal_validation_0917 as fv
    import validation_capture_0917 as cap
    import analyze_validation_0917 as ana

    model = fv.Path(fv.live._resolve_model_path())
    assert model.is_file()
    assert fv.sha256_file(model).lower() == fv.EXPECTED_MODEL_SHA256
    assert cap.VALIDATION_ROOT.parts[-2:] == ("Data", "Validation")
    assert ana.MODEL.is_file()
    print("FORMAL_VALIDATION_ENTRY_OK")

def analysis() -> None:
    import analyze_validation_0917 as ana
    import analyze_validation_gui_0917 as gui
    assert ana.MODEL.is_file()
    assert gui.VAL.parts[-2:] == ("Data", "Validation")
    assert gui.MIDI.parts[-2:] == ("Data", "Validation_MIDI")
    print("ANALYSIS_ENTRY_OK")
def alignment() -> None:
    import midi_csv_alignment_gui_0917_ZH_TW as gui

    gui.self_test()
    assert gui.DEFAULT_CSV_ROOT.parts[-2:] == ("Data", "CSV")
    assert gui.DEFAULT_MIDI_ROOT.parts[-2:] == ("Data", "MIDI")
    assert gui.DEFAULT_OUTPUT_ROOT.parts[-2:] == ("Data", "Alignment")
    print("ALIGNMENT_ENTRY_OK")


def main() -> int:
    actions = {"collector": collector, "model": model, "formal": formal, "analysis": analysis, "alignment": alignment}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        print("usage: smoke_test_entrypoints.py collector|model|formal|analysis|alignment")
        return 2
    actions[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



