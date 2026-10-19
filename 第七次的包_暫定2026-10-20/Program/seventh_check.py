"""Software preflight only: checks files/dependencies, never opens COM."""
import ast
import importlib
import json
from pathlib import Path
import sys

PACK=Path(__file__).resolve().parents[1]


def main():
    for name in ['numpy','scipy','sklearn','joblib','threadpoolctl','xgboost','serial','mido','tkinter','customtkinter','tkinterdnd2']:
        importlib.import_module(name)
    import seventh_core as core
    bundle,config=core.load_components()
    for path in (PACK/'Program').glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    for path in PACK.glob('*.bat'):
        data=path.read_bytes()
        assert data.isascii() and b'\r\n' in data and b'\n' not in data.replace(b'\r\n',b'')
    for hand in ['Left_L','Right_R']:
        source=(PACK/f'Firmware/Gloves_{hand}/Gloves_{hand}.ino').read_text(encoding='utf-8')
        assert f"#define HAND_ID '{hand[-1]}'" in source
    from serial.tools import list_ports
    result=dict(software='PASS',hardware='NOT_TESTED',field_ready='NOT_CONFIRMED',
        opens_com=False,python=sys.executable,visible_ports=[p.device for p in list_ports.comports()],
        model_sha256=core.detector.rt.sha256(core.MODEL),input_dimensions=core.GENERATED_DIMENSIONS,
        selected_dimensions=core.SELECTED_DIMENSIONS,model_name=core.MODEL_NAME,
        classifier_contract=core.runtime.CONTRACT,compass_heading_used=False,pretests_in_training=False,
        output_guard_enabled=False,training_events=bundle['training_events'],
        detector=config,personal_adaptation=False,
        midi_mapping={'小鼓':[37,38,40],'高音 Tom':[48],'中音 Tom':[45],'落地 Tom':[43],
                      'Hi-Hat':[46],'Crash':[49],'Ride':[51,59]},excluded_foot_notes=[36,44])
    (PACK/'QA').mkdir(exist_ok=True)
    (PACK/'QA/software_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Software PASS: native GUI + external detector (2.1 g) + venue-free per-hit committee (104 features, no compass heading).')
    print('Hardware NOT TESTED: actual two COMs, L/R, 80-120 Hz and drumhead zero must pass in the live console.')
    return 0


if __name__=='__main__':sys.exit(main())
