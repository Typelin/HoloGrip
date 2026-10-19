"""Build reproducible integrity metadata; never copies or changes field data."""
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path

import joblib

from fourth_song03_raw_runtime_1008 import sha256

PACK=Path(__file__).resolve().parents[1]


def main():
    model=PACK/'Model/venue_free_committee_1020.joblib';config=PACK/'Model/detector_venue_free_2p1g.json'
    assert sha256(model)=='c43c8d0f99510e97cf66f563558b7750ce579f40ff6e63f57f729ba887b7102d'
    bundle=joblib.load(model)
    assert bundle['contract']=='hologrip_venue_free_committee_1020_v1'
    assert not bundle['pretests_in_training'] and not bundle['compass_heading_used'] and bundle['per_hit_independent']
    assert bundle['detector_config']==json.loads(config.read_text(encoding='utf-8'))['config']
    assert len(bundle['portable'])==6
    files=[p for p in PACK.rglob('*') if p.is_file() and p.suffix in ['.py','.html','.ino','.bat']
        and p.relative_to(PACK).parts[0] not in ['Data','QA','今天']]
    hashes={p.relative_to(PACK).as_posix():sha256(p) for p in files}
    deps=['numpy','scipy','scikit-learn','joblib','threadpoolctl','xgboost','pyserial','mido','customtkinter','tkinterdnd2']
    versions={name:importlib.metadata.version(name) for name in deps}
    manifest=dict(schema='hologrip-seventh-field-package-v3',target_date='2026-10-20',prepared_date='2026-10-20',
        workflow_revision='2026-10-20-venue-free-committee',created_at=datetime.now().astimezone().isoformat(),
        model=dict(file=model.name,sha256=sha256(model),input_dimensions=104,selected_dimensions=104,
                   training_scope='songs+four-drum+clear single-hand of rounds 4/5/6; pretests excluded',
                   training_events=6667,mode='per_hand_committee',runtime_contract=bundle['contract'],
                   members=bundle['members'],learners=['xgboost_cpu_inference','mlp_numpy_forward'],
                   compass_heading_used=False,pretests_in_training=False,per_hit_independent=True,
                   future_context_ms=80,frozen=True),
        detector=dict(file=config.name,sha256=sha256(config),profile='venue_free_2p1g',
                      threshold_g=2.1,embedded_model_detector_ignored=False),
        output_guard=dict(enabled=False,reason='Every released hit is reported; no abstention or margin filter.'),
        runtime_hashes=hashes,dependencies=versions,
        architecture='external hit timing (2.1 g) -> heading-free trajectory summary + rotation relative to recorded drumhead zero -> per-hand XGBoost+MLP committee',
        research_reference=dict(pretests_in_training=False,dev_selected_on_pretests=True,unseen_day_blind_test=False,
            pretest_scores=[dict(round=4,correct=207,total=287),dict(round=5,correct=122,total=160),dict(round=6,correct=112,total=140)]),
        inactive_legacy_files=['Model/world_motion_456_clear219.joblib','Model/heading_relative_pose261.joblib','Model/heading_profile.json','Model/detector_conservative.json','Model/trajectory_456_348.joblib','Model/precision_guard.json','Program/seventh_heading_runtime.py'],
        workflow=dict(mode='direct_frozen_model_test',interface='native_tkinter',gui_entry='Program/seventh_native_gui.py',demonstration_required=False,
            expected_targets_used_during_inference=False,personal_adaptation=False,
            test_routine='Existing per-drum 20-tap test; no enforced order or count gate.'),
        midi_read_during_inference=False,hardware_validation='NOT_PERFORMED',software_validation='SEE_QA',
        formal_data_files_at_preparation=len([p for p in (PACK/'Data').rglob('*') if p.is_file()]))
    (PACK/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (PACK/'Program/requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(hashes),model_sha256=sha256(model),config_sha256=sha256(config)),ensure_ascii=False))


if __name__=='__main__':main()
