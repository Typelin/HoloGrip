"""Final model for the 8th package (10/23): rounds 4-7, all 13,617 training hits (pretests excluded),
XGBoost+MLP per hand, 6 seeds, portable format. Stores the per-hand per-drum training means of the
zero-relative rotation block for the causal per-drum offset correction."""
import numpy as np, joblib
import common as K
from final_model import portable_from

ROT = slice(83, 104)


def main():
    D = K.build(); rows = D['rows']
    com = K.train(rows, {}, seeds=tuple(range(6)))
    old = joblib.load(K.P7/'Model/venue_free_committee_1020.joblib')
    tm = {f'{h}|{d}': np.mean([r['x'][ROT] for r in rows if r['hand'] == h and r['drum'] == d], 0).astype(np.float32)
          for h in 'LR' for d in K.DRUMS if any(r['hand'] == h and r['drum'] == d for r in rows)}
    bundle = dict(old, contract='hologrip_venue_free_committee_1023_v1', portable=portable_from(com), members=[['all', s] for s in range(6)],
                  classes=[str(c) for c in com[0]['L'][0]], training_events=len(rows),
                  training='rounds 4-7: round4-6 reference labels, round7 songs (human + v2 auto labels), round7 protocol single-hand; pretests excluded',
                  drum_rotation_means=tm,
                  adaptation=dict(kind='per_drum_prototype_offset', alpha=1.0, confidence=0.7, min_confident_hits=20, passes=2,
                                  rot_slice=[83, 104], causal='past hits of the same hand in the take only'),
                  cymbal_rescue=dict(detector_threshold_g=1.7, strong_g=2.1, cymbals=['Hi-Hat', 'Crash', 'Ride'], min_score=0.6),
                  pretests_in_training=False, compass_heading_used=False, per_hit_independent=False, future_context_ms=80)
    out = K.P7.parent/'第八次的包_暫定2026-10-27/Model/venue_free_committee_1023.joblib'
    joblib.dump(bundle, out, compress=3); print('saved', out, len(rows))


if __name__ == '__main__':
    main()
