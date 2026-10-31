import json, numpy as np, itertools
from scipy.stats import rankdata, spearmanr
S='<scratch>'
taus=['0p00','0p10','0p20','0p35','0p50']; seeds=[42,43,44]
R={t:{s:json.load(open(f'{S}/results-ro/mjx_ppo_{t}_seed{s}/results.json')) for s in seeds} for t in taus}
bodies=sorted(R['0p00'][42])
aud=json.load(open(f'{S}/main-placebo/audit.json'))['bodies']
PR=json.load(open('train_like_probe_all_20.json'))
inert10=np.array([aud[b]['geometric_inert']['0.10'] for b in bodies])
inert20=np.array([aud[b]['geometric_inert']['0.20'] for b in bodies])
adv_reach=np.array([aud[b]['probe_min_torso_z']['min']<aud[b]['threshold']['0.10'] for b in bodies])
tl_reach=np.array([b in PR and (PR[b]['gauss']['eps_below']>0 or PR[b]['bang']['eps_below']>0) for b in bodies])
gauss_reach=np.array([b in PR and PR[b]['gauss']['eps_below']>0 for b in bodies])
L=np.array([[R['0p50'][s][b]['eval_tau050']['mean_ep_len'] for s in seeds] for b in bodies]); cut23=(L<=20).any(1)
fam=np.array([aud[b]['family'] for b in bodies])
def v(t,s,ev='eval_tau020'): return np.array([R[t][s][b][ev]['per_step_reward'] for b in bodies])
M={t:np.mean([v(t,s) for s in seeds],0) for t in taus}
def fast_rho(X,Y):  # rows = bootstrap draws
    rx=rankdata(X,axis=1); ry=rankdata(Y,axis=1)
    rx-=rx.mean(1,keepdims=True); ry-=ry.mean(1,keepdims=True)
    return (rx*ry).sum(1)/np.sqrt((rx**2).sum(1)*(ry**2).sum(1))
rng=np.random.default_rng(11)
contr=[('0p00','0p50','RF-Strict'),('0p20','0p50','Std-Strict'),('0p20','0p00','Std-RF'),('0p00','0p35','RF-0.35'),('0p10','0p50','0.10-Strict')]
def sb(r): return 3*r/(1+2*r)
def report(name,m,B=4000):
    idx=np.where(m)[0]; n=len(idx); I=idx[rng.integers(0,n,(B,n))]
    pl=spearmanr(M['0p00'][idx],M['0p10'][idx])[0]; bp=fast_rho(M['0p00'][I],M['0p10'][I])
    rAB=np.mean([spearmanr(v('0p00',a)[idx],v('0p10',b)[idx])[0] for a,b in itertools.permutations(seeds,2)])
    rAA=np.mean([spearmanr(v('0p00',a)[idx],v('0p00',b)[idx])[0] for a,b in itertools.combinations(seeds,2)])
    rBB=np.mean([spearmanr(v('0p10',a)[idx],v('0p10',b)[idx])[0] for a,b in itertools.combinations(seeds,2)])
    same=np.mean([spearmanr(v('0p00',s)[idx],v('0p10',s)[idx])[0] for s in seeds])
    dec=3*rAB/np.sqrt((1+2*rAA)*(1+2*rBB))
    fc={f:int((fam[idx]==f).sum()) for f in ['floor','vt','mvt']}
    print(f'== {name}: n={n} {fc}')
    print(f'   placebo 3-seed rho={pl:.3f} [{np.percentile(bp,2.5):.2f},{np.percentile(bp,97.5):.2f}]; single-run same-seed={same:.3f} cross-seed={rAB:.3f}; seed-decoupled estimate={dec:.3f}')
    for a,b,lab in contr:
        o=spearmanr(M[a][idx],M[b][idx])[0]; bo=fast_rho(M[a][I],M[b][I]); d=bp-bo
        ra=np.mean([spearmanr(v(a,x)[idx],v(a,y)[idx])[0] for x,y in itertools.combinations(seeds,2)])
        rb=np.mean([spearmanr(v(b,x)[idx],v(b,y)[idx])[0] for x,y in itertools.combinations(seeds,2)])
        ceil=np.sqrt(sb(ra)*sb(rb))
        print(f'   {lab:12s} rho={o:.3f} [{np.percentile(bo,2.5):.2f},{np.percentile(bo,97.5):.2f}]  placebo-minus CI [{np.percentile(d,2.5):.2f},{np.percentile(d,97.5):.2f}]  reliability ceiling={ceil:.2f}')
safe=inert10&~adv_reach&~tl_reach
print('counts: below radius',inert10.sum(),'| adversarial reach',(inert10&adv_reach).sum(),'| training-like reach',(inert10&tl_reach).sum(),'| gaussian reach',(inert10&gauss_reach).sum(),'| union',(inert10&(adv_reach|tl_reach)).sum(),'| safe',safe.sum())
report('A. 94 below torso radius (lead)',inert10)
report('B. 94 minus 5 bodies crossing under Gaussian actions',inert10&~gauss_reach)
report('C. safe under both probes',safe)
report('D. lead cleanest 59',inert10&~inert20&~cut23)
report('E. safe & Standard fires & no cut-off seed',safe&~inert20&~cut23)
