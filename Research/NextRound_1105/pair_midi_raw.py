"""Suggest which MIDI file belongs to which Raw recording of a field day (read only).

For every MIDI x Raw pair the MIDI note times are aligned to the glove peaks (autolabel_1027.align:
histogram of time differences, then 5 ms search and median refine); the share of notes that land
within 50 ms of a glove peak is the match rate. Each MIDI gets its best Raw; a close second best is
flagged. Also proposes the rounds.json entries (single-hand: dominant non-snare drum; songs: kind by
number of drums). The proposal must be checked by a person before labelling.

  python pair_midi_raw.py R9            -> out/pairing_R9.md / .json
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAY = sys.argv[1] if len(sys.argv) > 1 else 'R9'   # read before the research imports
sys.path[:0] = [str(ROOT/'Research/AutoLabel_1027'), str(HERE)]
import autolabel_1027 as AL

OUT = HERE/'out'; OUT.mkdir(exist_ok=True)


def main():
    cfg = json.loads((HERE/os.environ.get('HOLOGRIP_ROUNDS', 'rounds.json')).read_text(encoding='utf-8'))[DAY]
    pack = ROOT/cfg['package']/'Data'
    raws = sorted((pack/'Raw').glob('*_raw_100hz_*.csv'))+sorted((pack/'模型實測').glob(f"*/*_raw_100hz_{cfg['date']}_*.csv"))
    mids = sorted(Path(cfg['midi_dir']).glob('*.mid'))
    print('raw', len(raws), 'midi', len(mids), flush=True)
    peaks = {}
    for r in raws:
        sig = AL.A.signals(AL.A.rt.world.read_raw(r))
        peaks[r] = np.sort([p['t'] for p in sig['peaks'] if p['value'] >= .5])
    res = []
    for m in mids:
        ev = [e for e in AL.legacy.midi(m)['events'] if e['vel'] >= AL.GHOST_VEL]
        if len(ev) < 5:
            res.append(dict(midi=m.stem, notes=len(ev), best=None)); continue
        T = np.array([e['t'] for e in ev])
        sc = sorted(((AL.align(T, peaks[r])[1], r) for r in raws if len(peaks[r]) >= 5), key=lambda x: -x[0])
        drums = Counter(e['name'] for e in ev)
        if m.stem[-2:] in (' L', ' R') and len(drums) <= 4:
            target = max((d for d in drums if d != '小鼓'), key=drums.get, default='小鼓')
            proposal = dict(section='single', value=[target, sc[0][1].stem])
        else:
            proposal = dict(section='songs', value=['song' if len(drums) >= 5 else 'four_drum', sc[0][1].stem])
        res.append(dict(midi=m.stem, notes=len(ev), drums=dict(drums), best=sc[0][1].stem, best_rate=round(sc[0][0], 3),
                        second=sc[1][1].stem if len(sc) > 1 else None, second_rate=round(sc[1][0], 3) if len(sc) > 1 else None,
                        pretest_recording='模型實測' in str(sc[0][1]), proposal=proposal,
                        check=bool(sc[0][0] < 0.85 or (len(sc) > 1 and sc[0][0]-sc[1][0] < 0.15))))
    lines = ['| MIDI | 音符 | 最像的 Raw | 吻合 | 第二像 | 吻合 | 要人工看 |', '|---|---:|---|---:|---|---:|---|']
    for r in res:
        if r['best'] is None: lines.append(f"| {r['midi']} | {r['notes']} | （音符太少） | | | | 是 |"); continue
        lines.append(f"| {r['midi']} | {r['notes']} | {r['best']} | {r['best_rate']} | {r['second']} | {r['second_rate']} | {'是' if r['check'] else ''} |")
    md = '\n'.join(lines)
    (OUT/f'pairing_{DAY}.md').write_text(md+'\n', encoding='utf-8')
    (OUT/f'pairing_{DAY}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(md, flush=True)


if __name__ == '__main__':
    main()
