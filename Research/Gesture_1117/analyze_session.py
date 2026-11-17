"""Offline analysis of one gesture-lab recording: rule results per labelled segment + trajectory plots.

Usage: python analyze_session.py Data/<session> [--poses-from-data]
  --poses-from-data  when no pose calibration was saved: 'up' = median sensor-frame gravity of the still samples
                     in the 揮/③ segments, 'side' = same for the ② segment (per hand).
Writes <session>/analysis.png and <session>/analysis.txt (inputs read-only)."""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gesture_rules as GR


def load(session):
    rows = list(csv.DictReader(open(session/'gesture_raw.csv', encoding='utf-8-sig')))
    try: segs = json.loads((session/'segments.json').read_text(encoding='utf-8'))
    except Exception: segs = []
    return rows, segs


def poses_from_data(rows):
    poses = {}
    for h in 'LR':
        P = {}
        for which, labs in (('up', (f'{h}1', f'{h}3')), ('side', (f'{h}2',))):
            g = [GR.gravity_of(float(r['yaw_deg']), float(r['pitch_deg']), float(r['roll_deg'])) for r in rows
                 if r['hand'] == h and r['label'] in labs
                 and abs(np.linalg.norm([float(r['ax_g']), float(r['ay_g']), float(r['az_g'])])-1) < 0.04]
            if g: P[which] = GR.unit(np.median(np.array(g), 0)).tolist()
        if P: poses[h] = P
    return poses


def poses_from_strikes(rows, hand):
    """'side' / 'up' = median pre-strike pose of the strikes in the ② / ③ segments (what the GUI button does)."""
    eng = GR.HandRules(hand, {'pose_max_deg': 180}, {}); pre = {2: [], 3: []}
    for r in rows:
        if r['hand'] != hand: continue
        e, _ = eng.push(float(r['sensor_time_ms']), float(r['ax_g']), float(r['ay_g']), float(r['az_g']),
                        float(r['yaw_deg']), float(r['pitch_deg']), float(r['roll_deg']))
        g = int(r['label'][1]) if r['label'][:1] == hand and r['label'][1:] in ('2', '3') else 0
        if e and g and e['kind'] == 'strike' and e['d_tilt'] >= eng.strike_min_deg(): pre[g].append(e['pre_gvec'])
    if len(pre[2]) < 5 or len(pre[3]) < 5: return {}
    return dict(side=GR.unit(np.median(pre[2], 0)).tolist(), up=GR.unit(np.median(pre[3], 0)).tolist())


def signals(rows, hand):
    eng = GR.HandRules(hand); S = []
    for r in rows:
        if r['hand'] != hand: continue
        _, s = eng.push(float(r['sensor_time_ms']), float(r['ax_g']), float(r['ay_g']), float(r['az_g']),
                        float(r['yaw_deg']), float(r['pitch_deg']), float(r['roll_deg']))
        s['label'] = r['label']; S.append(s)
    return S


def main():
    session = Path(sys.argv[1]).resolve(); rows, segs = load(session)
    saved = HERE/'gesture_params.json'
    params, poses = {}, {}
    if saved.exists():
        d = json.loads(saved.read_text(encoding='utf-8')); params = d.get('params', {}); poses = d.get('poses', {})
    if '--poses-from-strikes' in sys.argv:
        poses = {h: poses_from_strikes(rows, h) for h in 'LR'}; poses = {h: v for h, v in poses.items() if v}; src = '從這次敲擊估計'
    elif '--poses-from-data' in sys.argv or not any(poses.get(h) for h in 'LR'):
        poses = poses_from_data(rows); src = '從資料估計'
    else:
        src = '已儲存的校正'
    evs = GR.run_file(rows, params, poses)
    lines = [f'錄音：{session.name}　姿勢：{src}', '']
    for h, P in poses.items():
        if 'up' in P and 'side' in P: lines.append(f'{h} 兩個姿勢相差 {GR.angle_deg(P["up"], P["side"]):.0f}°')
    labels = [s['label'] for s in segs] + ['']
    lines.append('')
    lines.append(f'{"段落":<8s}{"判出":<40s}')
    for lab in labels:
        cnt = Counter((e['name'] if e['gesture'] else '不算：'+e['reason'].split('（')[0]) for e in evs if e['label'] == lab)
        lines.append(f'{lab or "（段落外）":<8s}' + '、'.join(f'{k} {v}' for k, v in cnt.most_common()))
    lines.append('')
    for e in evs:
        lines.append(f'{e["label"] or "-":<4s} {e["hand"]} {e["kind"]:<6s} t={e["peak_t"]/1000:8.2f}s {e["name"]:<12s} '
                     f'水平 {e["d_yaw"]:+6.0f}° 往下 {e["d_tilt"]:5.0f}° 占比 {e["yaw_share"]:.2f} {e["peak_g"]:.1f} g 姿勢 {e["pose"]} {e["reason"]}')
    (session/'analysis.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:30]))
    plot(session, rows, segs, evs)


def plot(session, rows, segs, evs):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans TC', 'sans-serif']; plt.rcParams['axes.unicode_minus'] = False
    hands = [h for h in 'LR' if any(r['hand'] == h for r in rows)]
    segl = [(s, s['label'][0]) for s in segs if s['label'][0] in hands and s['label'][1:] in ('1', '2', '3')]
    segl += [(s, h) for s in segs if s['label'] == 'NOISE' for h in hands]
    sig = {h: signals(rows, h) for h in hands}
    fig, axes = plt.subplots(len(segl), 1, figsize=(15, 3.2*len(segl)), squeeze=False)
    for ax, (s, h) in zip(axes[:, 0], segl):
        S = [x for x in sig[h] if x['label'] == s['label'] and s['start_t_ms'] is not None]
        if not S: continue
        t = np.array([x['t'] for x in S])/1000; t0 = t[0]
        ax.plot(t-t0, [x['yaw_rate'] for x in S], color='#1f6fd1', lw=1, label='水平轉速 °/s（+＝往左）')
        ax.plot(t-t0, [x['tilt_rate'] for x in S], color='#e8820c', lw=1, label='往下／上轉速 °/s')
        ax2 = ax.twinx(); ax2.plot(t-t0, [x['acc'] for x in S], color='#2a9d4a', lw=1, label='加速度 g'); ax2.set_ylim(0, 8)
        ax2.axhline(GR.DEFAULT_PARAMS['strike_g'], color='#2a9d4a', ls=':', lw=1)
        for e in evs:
            if e['label'] == s['label'] and e['hand'] == h:
                x = e['peak_t']/1000-t0; ok = e['gesture']
                ax.axvline(x, color='#0a6' if ok else '#c33', lw=1.2, alpha=.7)
                ax.text(x, ax.get_ylim()[1]*0.92 if ok else ax.get_ylim()[1]*0.78, str(e['gesture']) if ok else '×',
                        color='#0a6' if ok else '#c33', ha='center', fontsize=9)
        ax.set_title(f'{s["text"]}｜{"左" if h == "L" else "右"}手（綠線＝判出，數字＝手勢編號；紅×＝不算）', loc='left', fontsize=11)
        ax.set_ylabel('°/s'); ax2.set_ylabel('g'); ax.set_xlabel('秒')
        if (s, h) == segl[0]:
            l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels(); ax.legend(l1+l2, n1+n2, loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(session/'analysis.png', dpi=90)


if __name__ == '__main__':
    main()
