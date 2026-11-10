import json, sys
b = json.load(open('out/guard_base.json', encoding='utf-8'))
for tag in sys.argv[1:]:
    d = json.load(open(f'out/guard_{tag}.json', encoding='utf-8'))
    print('##', tag, {k: (v['ok'], v['rate'], v['cells_failed']) for k, v in d['summary'].items()})
    for k in d['results']:
        for n, v in sorted(d['results'][k].items()):
            o = b['results'][k][n]
            if v['ok'] != o['ok']: print('  ', k, n, o['ok'], '->', v['ok'], f"({v['ok']-o['ok']:+d})", f"{o['rate']:.3f}->{v['rate']:.3f}")
