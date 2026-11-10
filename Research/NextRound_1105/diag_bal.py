import sys, os
os.environ['HG_STATS'] = 'off'
sys.argv = ['x']
import guard_eval as G, anchor_guard as AG
from collections import Counter
J = {(j[0], j[1]): j for j in G.jobs()}
pick = [('takes_full', 'R10:歌曲2'), ('takes_lodo', 'seventh_06')]
def conf(job):
    kind, name, model, raw, truth = job
    P, rt = G.stream(model, raw); pairs = G.VP.V.match(truth, P)
    errs = Counter((g['hand'], g['drum'], P[pairs[i]]['drum'] if i in pairs else 'miss') for i, g in enumerate(truth) if not (i in pairs and P[pairs[i]]['drum'] == g['drum'] and P[pairs[i]]['hand'] == g['hand']))
    tot = Counter((g['hand'], g['drum']) for g in truth)
    return errs, tot, rt
base = {k: conf(J[k]) for k in pick}
AG.install_stats(G.core.runtime, balance=10, min_n=5)
for k in pick:
    e1, tot, rt = conf(J[k]); e0 = base[k][0]
    print('##', k, 'truth', dict(tot))
    print('  base', e0.most_common(6)); print('  bal ', e1.most_common(6))
    for h in 'LR':
        print('  stats n', h, dict(zip([c[:2] for c in rt.classes], rt.stats[h][1].n.astype(int))))
