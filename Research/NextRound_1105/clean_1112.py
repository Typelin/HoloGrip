"""11/12 (person 2) cleanliness check of the two pretests and song 1, then success rates on clean cells only.

Per hand x drum cell: yaw/pitch at each hit (field engine replay, 11-12 package) relative to the take's own snare,
spread (IQR) and distance from the same person's controlled-drill position (L/R_H_M_R, L/R_M_R_G, from pose_1112).
A cell is "deviation too large" when
  (a) scattered: circular yaw IQR > 25 deg (arm/body swaying while playing), or
  (b) on another drum's spot: its median yaw is within 10 deg (pitch within 15 deg) of a drum the same person keeps
      >= 25 deg apart in the controlled drills (same hand, same take); of the two cells the one farther from its
      controlled position is excluded.
Scores: live judgement written in the field (model e1) and a replay through 11-17的包 (model g6; pretests only,
the song is in g6's training data). Song cells use single notes only (no other note within 40 ms).
Read-only: writes out/clean_1112.txt and out/clean_1112.json."""
import csv, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
PKS = {'e1': ROOT/'11-12的包', 'g6': ROOT/'11-17的包'}; PK = PKS['e1']
M = PK/'今天'/'雲端_11-12'
sys.path[:0] = [str(HERE), str(ROOT/'Research/AutoLabel_1027')]
TAKES = [('前測 正打', 'pre:112540', '11：12.mid'), ('前測 反打', 'pre:113018', '11：12 測試.mid'), ('歌曲1', '113853', '11：12歌曲1(修正.mid')]
# controlled-drill yaw vs own snare (pose_1112: mean of the controlled takes that contain the drum)
REF = {('L', '高音 Tom'): -44, ('L', '中音 Tom'): -51.5, ('L', 'Ride'): -83, ('L', '落地 Tom'): -68,
       ('R', '高音 Tom'): -38, ('R', '中音 Tom'): -66.5, ('R', 'Ride'): -102.5, ('R', '落地 Tom'): -110}
IQR_MAX, NEAR_YAW, NEAR_PITCH = 25.0, 10.0, 15.0
# drum pairs the same person keeps >= 25 deg apart (yaw) in the controlled drills of the same take
DISTINCT = {('R', '中音 Tom', 'Ride'), ('R', '中音 Tom', '落地 Tom'), ('L', '中音 Tom', '落地 Tom'),
            ('L', '高音 Tom', 'Ride'), ('R', '高音 Tom', 'Ride'), ('L', '高音 Tom', '落地 Tom'), ('R', '高音 Tom', '落地 Tom')}


def circ(deg):
    return float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(deg))))))


def raw_of(key):
    if key.startswith('pre:'):
        return next(p for p in (PK/'Data/模型實測').glob(f'*_{key[4:]}_*/*_raw_100hz_*.csv'))
    return next((PK/'Data/Raw').glob(f'*_{key}.csv'))


def live_of(key, raw):
    f = raw.parent/'predictions.csv' if key.startswith('pre:') else raw.parent/'即時判斷'/f'{raw.stem}_即時判斷.csv'
    return [dict(hand=r['hand'], t=float(r['peak_ms']), peak_ms=float(r['peak_ms']), drum=r['drum'])
            for r in csv.DictReader(open(f, encoding='utf-8-sig')) if r['accepted'] in ('True', '1')]


def truth_of(raw, mid, song):
    import rounds_data as RD, autolabel_1027 as AL
    cand = RD.L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]; ct = np.array(sorted(x['t'] for x in cand))
    ev = [e for e in AL.legacy.midi(mid)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
    mt = np.array([e['t'] for e in ev])
    grid = np.arange(0.88, 1.16, 0.0005) if song else np.arange(0.95, 1.15, 0.0005)
    s, off, rate = max(((s,)+tuple(AL.align(mt*s, ct)) for s in np.r_[1.0, grid]), key=lambda x: x[2])
    out = []
    for e in ev:
        t = e['t']*s+off; near = [x for x in cand if abs(x['t']-t) <= 60]
        out.append(dict(id=e['id'], t=t, drum=e['name'], hand=max(near, key=lambda x: x['g'])['hand'] if near else None))
    if song:   # simultaneous notes: the hand key is unreliable, keep the hand only on single notes
        ts = np.array([g['t'] for g in out])
        for g in out:
            if int(np.sum(np.abs(ts-g['t']) <= 40)) > 1: g['hand'] = None
    return out, rate


def replay(job):
    tag, key = job
    sys.path.insert(0, str(PKS[tag]/'Program'))
    import seventh_core as core
    raw = raw_of(key); b, config = core.load_components()
    hands = core.trajectory.read_raw(str(raw))
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    anchor = core.runtime.PRETEST_SNARE_ANCHOR if key.startswith('pre:') else 5
    engine = core.Engine(b, config, snare_anchor=anchor); rt = engine.runtime; rec = []; orig = rt._classify
    def spy(hand, x):
        rec.append(core.runtime._yaw_pitch(x).tolist()); return orig(hand, x)
    rt._classify = spy; out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; n0 = len(rec)
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        for q, yp in zip(p, rec[n0:]): out.append(dict(hand=q['hand'], t=q['peak_ms'], peak_ms=q['peak_ms'], drum=q['drum'], yaw=yp[0], pitch=yp[1]))
    return tag, key, out


def main():
    from verify_package_1027 import score
    jobs = [('e1', k) for _, k, _ in TAKES]+[('g6', k) for _, k, _ in TAKES if k.startswith('pre:')]
    with Pool(5, maxtasksperchild=1) as pool: REP = {(t, k): o for t, k, o in pool.imap(replay, jobs)}
    lines = [__doc__.split('\n')[0], f'rule: circular yaw IQR > {IQR_MAX:.0f} deg, or within {NEAR_YAW:.0f} deg yaw / {NEAR_PITCH:.0f} deg pitch of a drum kept apart in controlled drills -> excluded', '']
    res = {}
    for name, key, mid in TAKES:
        raw = raw_of(key); song = not key.startswith('pre:')
        truth, rate = truth_of(raw, M/mid, song); live = live_of(key, raw); rep = REP[('e1', key)]
        pose = defaultdict(list)
        for g in truth:
            if not g['hand']: continue
            near = [o for o in rep if o['hand'] == g['hand'] and abs(o['t']-g['t']) <= 60]
            if near: o = min(near, key=lambda o: abs(o['t']-g['t'])); pose[(g['hand'], g['drum'])].append((o['yaw'], o['pitch'], g['t']))
        geo = {}
        for h in 'LR':
            sn = np.array(pose.get((h, '小鼓'), [(0, 0, 0)]))
            ry = np.degrees(np.angle(np.mean(np.exp(1j*np.radians(sn[:, 0]))))); rp = float(np.median(sn[:, 1]))
            for (hh, d), v in pose.items():
                if hh != h: continue
                a = np.array(v); w = (a[:, 0]-ry+180) % 360-180; my = circ(w); c = (w-my+180) % 360-180
                q75, q25 = np.percentile(c, [75, 25]); ref = REF.get((h, d))
                reasons = []
                if len(v) >= 5 and q75-q25 > IQR_MAX: reasons.append(f'分散 ±{q75-q25:.0f}°')
                geo[f'{h}|{d}'] = dict(yaw=my, pitch=float(np.median(a[:, 1]))-rp, iqr=float(q75-q25), ref=ref, n=len(v), reasons=reasons)
        for h, d1, d2 in DISTINCT:
            g1, g2 = geo.get(f'{h}|{d1}'), geo.get(f'{h}|{d2}')
            if not g1 or not g2 or g1['n'] < 5 or g2['n'] < 5: continue
            if abs(g1['yaw']-g2['yaw']) < NEAR_YAW and abs(g1['pitch']-g2['pitch']) < NEAR_PITCH:
                moved, other = (g1, d2) if abs(g1['yaw']-g1['ref']) >= abs(g2['yaw']-g2['ref']) else (g2, d1)
                moved['reasons'].append(f'打在{other.split()[0]}的位置（差 {abs(g1["yaw"]-g2["yaw"]):.0f}°）')
        bad = {k for k, g in geo.items() if g['reasons']}
        r = dict(align=rate, cells={}, bad=sorted(bad))
        for tag in (['e1', 'g6'] if not song else ['e1']):
            P = live if tag == 'e1' else REP[('g6', key)]
            ok, n, cells = score(truth, P)
            if song:   # hand+drum on single notes
                n = sum(c[1] for c in cells.values()); ok = sum(c[0] for c in cells.values())
            cok = sum(c[0] for k, c in cells.items() if k not in bad); cn = sum(c[1] for k, c in cells.items() if k not in bad)
            if not song:   # notes without a hand key (none expected in pretests) stay in the clean total
                cok += ok-sum(c[0] for c in cells.values()); cn += n-sum(c[1] for c in cells.values())
            r[tag] = dict(ok=ok, n=n, clean_ok=cok, clean_n=cn, cells={k: c[:2] for k, c in cells.items()},
                          fail=[k for k, c in cells.items() if k not in bad and c[1] >= 5 and c[0]/c[1] < 0.85])
        res[name] = r
        lines.append(f'== {name}  ({raw.name}, MIDI {mid}, align {rate:.3f})')
        for tag in (['e1', 'g6'] if not song else ['e1']):
            x = r[tag]; lab = '現場即時 e1' if tag == 'e1' else '11-17包重播 g6'
            lines.append(f'   {lab}: 全部 {x["ok"]}/{x["n"]} = {x["ok"]/x["n"]:.1%}   乾淨格 {x["clean_ok"]}/{x["clean_n"]} = {x["clean_ok"]/x["clean_n"]:.1%}')
        for k in sorted(geo, key=lambda k: (k[0], k)):
            g = geo[k]; c1 = r['e1']['cells'].get(k, [0, 0]); c2 = r.get('g6', {}).get('cells', {}).get(k)
            lines.append(f'     {k:12s} n{g["n"]:4d} yaw {g["yaw"]:+5.0f} (ref {g["ref"] if g["ref"] is not None else "-":>6}) ±{g["iqr"]:3.0f}  e1 {c1[0]:3d}/{c1[1]:3d}'
                         + (f'  g6 {c2[0]:3d}/{c2[1]:3d}' if c2 else '') + ('   <- 不算：'+'、'.join(g['reasons']) if g['reasons'] else ''))
        lines.append('')
    txt = '\n'.join(lines); print(txt)
    (HERE/'out/clean_1112.txt').write_text(txt, encoding='utf-8')
    (HERE/'out/clean_1112.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
