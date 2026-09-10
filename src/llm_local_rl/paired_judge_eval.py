"""Paired probabilistic evaluation with explicit A/B referent alignment."""
from __future__ import annotations
import math


def calibration_metrics(rows: list[dict], *, bins: int = 10) -> dict:
    """Binary Brier; confidence ECE; target=0.5 ties preserved separately."""
    if bins < 1:
        raise ValueError('bins must be positive')
    for row in rows:
        if not all(math.isfinite(float(row[k])) and 0 <= float(row[k]) <= 1
                   for k in ('p_a', 'target', 'p_forward_a', 'p_reverse_referent_a')):
            raise ValueError('Invalid probability or target')
    def measure(group):
        if not group:
            return {'n': 0, 'brier': None, 'log_loss': None, 'ece': None, 'accuracy': None, 'bins': []}
        buckets=[[] for _ in range(bins)]
        for row in group:
            p=float(row['p_a']);y=float(row['target']);confidence=max(p,1-p)
            correctness=y if p>=.5 else 1-y
            buckets[min(bins-1,int(confidence*bins))].append((confidence,correctness))
        reliability=[]
        for index,bucket in enumerate(buckets):
            n=len(bucket)
            reliability.append({'lower':index/bins,'upper':(index+1)/bins,'n':n,
                                'confidence':sum(x for x,y in bucket)/n if n else None,
                                'accuracy':sum(y for x,y in bucket)/n if n else None})
        return {'n':len(group),
                'brier':sum((r['p_a']-r['target'])**2 for r in group)/len(group),
                'log_loss':-sum(r['target']*math.log(max(r['p_a'],1e-12))+(1-r['target'])*math.log(max(1-r['p_a'],1e-12)) for r in group)/len(group),
                'accuracy':sum(r['target'] if r['p_a']>=.5 else 1-r['target'] for r in group)/len(group),
                'ece':sum(b['n']*abs(b['confidence']-b['accuracy']) for b in reliability if b['n'])/len(group),
                'bins':reliability}
    clear=[r for r in rows if r['target']!=.5]
    ties=[r for r in rows if r['target']==.5]
    return {'all_targets':measure(rows),'strict_preferences':measure(clear),'score_ties':measure(ties),
            'order_consistency_rate':sum((r['p_forward_a']>=.5)==(r['p_reverse_referent_a']>.5) for r in rows)/len(rows) if rows else None,
            'exact_order_probability_ties':sum(r['p_forward_a']==.5 or r['p_reverse_referent_a']==.5 for r in rows),
            'order_probability_gap_mean':sum(abs(r['p_forward_a']-r['p_reverse_referent_a']) for r in rows)/len(rows) if rows else None,
            'probability_definition':'sigmoid((forward_log_odds_A - reverse_log_odds_A)/2)',
            'brier_definition':'mean((p_A - target_A)^2); binary convention, not doubled',
            'ece_definition':'10 equal-width confidence bins; no post-hoc calibration fitted'}


def paired_brier_difference(active: list[dict], shadow: list[dict]) -> dict:
    a={r['id']:r for r in active};b={r['id']:r for r in shadow}
    if len(a)!=len(active) or len(b)!=len(shadow) or a.keys()!=b.keys():
        raise ValueError('Paired judges must cover the same unique panel IDs')
    diffs=[]
    for key in a:
        if a[key]['target']!=b[key]['target']:
            raise ValueError('Paired judges have different targets')
        diffs.append((b[key]['p_a']-a[key]['target'])**2-(a[key]['p_a']-a[key]['target'])**2)
    return {'n':len(diffs),'shadow_minus_active_brier':sum(diffs)/len(diffs) if diffs else None,
            'interpretation':'Negative favors random-initialized shadow; comparison conditional on active-judge training stream.'}
