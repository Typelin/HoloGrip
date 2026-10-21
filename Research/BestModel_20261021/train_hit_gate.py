"""Final hit gate for the 8th package: XGBoost on all non-pretest candidates (1.5 g detector),
features = peak g + detector shape (18) + 104 venue-free + context (other hand max |a| +-40 ms,
log time since previous same-hand candidate, g ratio to it). Stored in the 8th-package bundle."""
import json, sys
import numpy as np, joblib, xgboost as xgb
sys.argv = ['x', '1p5']
import hit_clf as H, common as K

PACK = K.P7.parent/'第八次的包_暫定2026-10-23'


def main():
    names = [n for n in H.CAND if n not in K.TESTS]
    X = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[H.labels(n)[0] != 2] for n in names])
    y = np.concatenate([H.labels(n)[0][H.labels(n)[0] != 2] for n in names])
    m = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0)
    m.fit(X, y); bst = m.get_booster(); bst.set_param({'device': 'cpu'})
    print('train rows', len(y), 'hits', int(y.sum()), 'dims', X.shape[1])
    f = PACK/'Model/venue_free_committee_1023.joblib'; b = joblib.load(f)
    b['hit_gate'] = dict(xgb_json=bytes(bst.save_raw('json')), threshold=0.4, dims=int(X.shape[1]),
                         features='peak_g, detector feature(18), venue-free x(104), ctx[other-hand max |a| +-40ms, log1p(ms since previous same-hand candidate, cap 3000), g / previous candidate g]',
                         training=f'{len(names)} non-pretest recordings, {len(y)} candidates; leave-one-day-out recall 98.8% precision 99.1%')
    b['detector_config'] = dict(b['detector_config'], threshold_g=1.5)
    b['cymbal_rescue'] = None
    joblib.dump(b, f, compress=3); print('saved')


if __name__ == '__main__':
    main()
