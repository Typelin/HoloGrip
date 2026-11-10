"""Relative-path launcher dispatch with real, non-ambiguous return codes."""
import os
from pathlib import Path
import subprocess
import sys

PACK=Path(__file__).resolve().parents[1]


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'check'
    args=sys.argv[2:]
    targets=dict(check='seventh_check.py',live='seventh_native_gui.py',collect='seventh_collect.py',
                 midi='midi_anomaly_check.py',align='midi_csv_alignment_gui.py',replay='seventh_replay.py')
    if mode=='data':os.startfile(str(PACK/'Data'));return 0
    if mode not in targets:raise ValueError('Unknown mode')
    if mode=='midi' and not args:args=['--choose']
    result=subprocess.run([sys.executable,str(PACK/'Program'/targets[mode]),*args],cwd=PACK)
    if mode=='midi' and result.returncode==1:
        print('MIDI REVIEW: report created. This is not a launcher crash.')
    return result.returncode


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(2)
