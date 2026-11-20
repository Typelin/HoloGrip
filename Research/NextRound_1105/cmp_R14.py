"""Compare 11/17 retrain candidates (out/score_pkg_<tag>.json, 11-17的包 runtime) against g6 (score_pkg_g6_R14.json).
Usage: python cmp_R14.py h1 h2 ..."""
import json, sys
from pathlib import Path
O = Path(__file__).resolve().parent/'out'
GROUPS = {'人員3 11/10': ['pretest:R12#1', 'pretest:R12#2', 'pretest:R12#3'], '人員3 11/17': ['pretest:R14'],
          '人員2 11/12': ['pretest:R13#1', 'pretest:R13#2']}


def L(t): return json.loads((O/f'score_{t}.json').read_text(encoding='utf-8'))


def summary(d):
    tk = d['takes']; ok = sum(v['ok'] for v in tk.values()); n = sum(v['n'] for v in tk.values())
    nb = {k: v for k, v in tk.items() if k != 'pretest:R10b'}
    fail = sum(1 for v in nb.values() for c in v['cells'].values() if c[1] >= 10 and c[0] < 0.85*c[1])
    g = {name: sum(tk[t]['ok'] for t in ts if t in tk) for name, ts in GROUPS.items()}
    return ok, n, sum(v['ok'] for v in nb.values()), sum(v['n'] for v in nb.values()), fail, g


if __name__ == '__main__':
    base = L('pkg_g6_R14'); tags = ['pkg_g6_R14']+[f'pkg_{t}' for t in sys.argv[1:]]
    print('| model | all | excl. R10b | cells <85% | '+' | '.join(GROUPS)+' |')
    for t in tags:
        try: d = L(t)
        except FileNotFoundError: continue
        ok, n, ok2, n2, fail, g = summary(d)
        print(f'| {t[4:]} | {ok}/{n} | {ok2}/{n2} | {fail} | '+' | '.join(str(g[k]) for k in GROUPS)+' |')
    for t in tags[1:]:
        try: d = L(t)
        except FileNotFoundError: continue
        print(f'-- {t[4:]} vs g6 per take:', ', '.join(f"{k.replace('pretest:', '')} {d['takes'][k]['ok']-v['ok']:+d}" for k, v in base['takes'].items() if d['takes'][k]['ok'] != v['ok']))
        ch = [f"{k.replace('pretest:', '')} {c} {v['cells'][c][0]}->{d['takes'][k]['cells'][c][0]}/{v['cells'][c][1]}"
              for k, v in base['takes'].items() for c in v['cells'] if abs(d['takes'][k]['cells'][c][0]-v['cells'][c][0]) >= 3]
        print('   cells moved >=3:', '; '.join(ch))
