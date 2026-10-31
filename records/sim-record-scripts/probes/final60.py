"""Archived probe step that chose the 60 bodies of the 0.10 control.

Run in the probe session's working directory after final_sets.py, whose loaders,
body sets and report() it reuses through exec, so its names are defined there.
"""
# ruff: noqa: F401, F821
import json, numpy as np, itertools
from scipy.stats import rankdata, spearmanr
exec(open('final_sets.py').read().split("safe=inert10")[0])  # reuse loaders, report(), fast_rho
NR=json.load(open('train_like_probe_near_100.json'))
near_reach=np.array([b in NR and (NR[b]['gauss']['eps_below']>0 or NR[b]['bang']['eps_below']>0) for b in bodies])
safe60=inert10&~adv_reach&~tl_reach&~near_reach
print('safe60 n =',safe60.sum(),'| excluded (any probe reaches 0.1 z0):',(inert10&~safe60).sum())
report('F. 60 bodies no probe reaches 0.1 z0',safe60)
report('G. F & Standard fires & no Strict cut-off seed',safe60&~inert20&~cut23)
# top-10 overlap for reference on F
idx=np.where(safe60)[0]
def top(a,b,k=10): return len(set(np.argsort(-a)[:k])&set(np.argsort(-b)[:k]))
print('top-10 overlap on F: placebo',top(M['0p00'][idx],M['0p10'][idx]),'RF-Strict',top(M['0p00'][idx],M['0p50'][idx]))
json.dump(sorted(np.array(bodies)[safe60].tolist()),open('placebo_safe60_bodies.json','w'),indent=0)
