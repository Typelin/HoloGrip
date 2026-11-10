"""11/10 songs: score the collector's live judgement (Raw/即時判斷/*.csv) against the song MIDI.
Answer key = MIDI notes aligned to the glove hit candidates, hand = strongest candidate within 60 ms
(same as the pretest key; no human hand labels yet). Each raw is paired with the MIDI that aligns best."""
import csv, json
from pathlib import Path
import rounds_data as RD
import verify_package_1027 as VP
L5 = RD.L5
PK = RD.ROOT/'11-10的包'
if __name__ == '__main__':
    raws = sorted((PK/'Data/Raw').glob('*_raw_100hz_*.csv')); mids = sorted((PK/'今天').glob('歌曲*.mid'))
    out = {}
    for raw in raws:
        live = raw.parent/'即時判斷'/f'{raw.stem}_即時判斷.csv'
        P = [dict(hand=r['hand'], t=float(r['peak_ms']), peak_ms=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(open(live, encoding='utf-8-sig')) if r['accepted'] in ('True', '1')]
        cand = L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]
        if len(cand) < 50 or len(P) < 20: print(raw.stem, 'skipped: only', len(cand), 'candidates /', len(P), 'live hits', flush=True); continue
        best = max(((m,)+RD.truth_of(m, cand) for m in mids), key=lambda x: x[2]['align_rate'])
        m, truth, info = best
        ok, n, cells = VP.score(truth, P)
        out[raw.stem] = dict(midi=m.name, align=info, live_hits=len(P), ok=ok, n=n, cells={k: [v[0], v[1], v[2]] for k, v in cells.items()})
        print(raw.stem, m.name, info['align_rate'], 'live', len(P), 'score', ok, '/', n, round(ok/n, 3), flush=True)
        for k, v in sorted(cells.items()):
            if v[0] < v[1]: print('    ', k, v[0], '/', v[1], v[2])
    (RD.OUT/'score_1110_live_songs.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
