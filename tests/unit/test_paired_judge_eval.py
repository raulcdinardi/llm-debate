import pytest
from llm_local_rl.paired_judge_eval import calibration_metrics, paired_brier_difference

def row(i,p,y,pf=None,pr=None):
 return dict(id=str(i),p_a=p,target=y,p_forward_a=p if pf is None else pf,p_reverse_referent_a=p if pr is None else pr)

def test_known_brier_order_alignment_and_ties():
 rows=[row(1,.8,1,.9,.7),row(2,.2,0,.1,.3),row(3,.5,.5,.8,.2)]
 m=calibration_metrics(rows)
 assert m['strict_preferences']['brier']==pytest.approx(.04)
 assert m['strict_preferences']['ece']==pytest.approx(.2)
 assert m['score_ties']['n']==1 and m['score_ties']['brier']==0
 assert m['order_consistency_rate']==pytest.approx(2/3)
 assert m['all_targets']['n']==3

def test_paired_difference_checks_identity_and_targets():
 a=[row(1,.8,1),row(2,.2,0)];b=[row(2,.1,0),row(1,.9,1)]
 assert paired_brier_difference(a,b)['shadow_minus_active_brier']==pytest.approx(-.03)
 with pytest.raises(ValueError):paired_brier_difference(a,b[:1])
 with pytest.raises(ValueError):paired_brier_difference(a,[row(1,.9,0),b[0]])

def test_exact_probability_ties_use_visual_a_tiebreak_in_each_order():
 m=calibration_metrics([row(1,.5,1,.5,.5)])
 assert m['order_consistency_rate']==0
 assert m['exact_order_probability_ties']==1
