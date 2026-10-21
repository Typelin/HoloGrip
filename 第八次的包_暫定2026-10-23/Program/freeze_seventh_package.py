"""Build reproducible integrity metadata; never copies or changes field data."""
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path

import joblib

from fourth_song03_raw_runtime_1008 import sha256

PACK=Path(__file__).resolve().parents[1]


def main():
    model=PACK/'Model/venue_free_committee_1023.joblib';config=PACK/'Model/detector_venue_free_1p7g.json'
    assert sha256(model)=='abfef111139b7ee318295df7678f61a1d742702014d02daff88684b4eced55e3'
    bundle=joblib.load(model)
    assert bundle['contract']=='hologrip_venue_free_committee_1023_v1'
    assert not bundle['pretests_in_training'] and not bundle['compass_heading_used'] and bundle['training_events']==13617
    assert bundle['detector_config']==json.loads(config.read_text(encoding='utf-8'))['config']
    assert len(bundle['portable'])==6
    files=[p for p in PACK.rglob('*') if p.is_file() and p.suffix in ['.py','.html','.ino','.bat']
        and p.relative_to(PACK).parts[0] not in ['Data','QA','今天']]
    hashes={p.relative_to(PACK).as_posix():sha256(p) for p in files}
    deps=['numpy','scipy','scikit-learn','joblib','threadpoolctl','xgboost','pyserial','mido','customtkinter','tkinterdnd2']
    versions={name:importlib.metadata.version(name) for name in deps}
    manifest=dict(schema='hologrip-eighth-field-package-v1',target_date='2026-10-23',prepared_date='2026-10-21',
        workflow_revision='2026-10-23-rounds4to7-adaptive',created_at=datetime.now().astimezone().isoformat(),
        model=dict(file=model.name,sha256=sha256(model),input_dimensions=104,selected_dimensions=104,
                   training_scope='rounds 4-7 songs, four-drum, single-hand; pretests excluded',
                   training_events=13617,mode='per_hand_committee',runtime_contract=bundle['contract'],
                   members=bundle['members'],learners=['xgboost_cpu_inference','mlp_numpy_forward'],
                   compass_heading_used=False,pretests_in_training=False,per_hit_independent=False,adaptation=bundle['adaptation'],cymbal_rescue=bundle['cymbal_rescue'],
                   future_context_ms=80,frozen=True),
        detector=dict(file=config.name,sha256=sha256(config),profile='venue_free_1p7g_cymbal_rescue',
                      threshold_g=1.7,embedded_model_detector_ignored=False),
        output_guard=dict(enabled=False,reason='Every released hit is reported; no abstention or margin filter.'),
        runtime_hashes=hashes,dependencies=versions,
        architecture='detector 1.7 g -> 104 venue-free features -> per-hand XGBoost+MLP committee -> causal per-drum offset correction -> cymbal rescue gate for peaks < 2.1 g',
        research_reference=dict(pretests_in_training=False,dev_selected_on_pretests=True,unseen_day_blind_test=False,
            pretest_scores='replay of 6 pretests through this engine: see 改動紀錄_第八次包_1023_ZH_TW.md'),
        inactive_legacy_files=['Program/seventh_heading_runtime.py'],
        workflow=dict(mode='direct_frozen_model_test',interface='native_tkinter',gui_entry='Program/seventh_native_gui.py',demonstration_required=False,
            expected_targets_used_during_inference=False,personal_adaptation=False,
            test_routine='Existing per-drum 20-tap test; no enforced order or count gate.'),
        midi_read_during_inference=False,hardware_validation='NOT_PERFORMED',software_validation='SEE_QA',
        formal_data_files_at_preparation=len([p for p in (PACK/'Data').rglob('*') if p.is_file()]))
    (PACK/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (PACK/'Program/requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(hashes),model_sha256=sha256(model),config_sha256=sha256(config)),ensure_ascii=False))


if __name__=='__main__':main()
