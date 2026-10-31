import json, sys
from multiprocessing import Pool
sys.argv=['x']
import train_like_probe as T
if __name__=='__main__':
    aud=json.load(open('<scratch>'))['bodies']
    near=json.load(open('near_margin_safe.json'))
    with Pool(14) as p:
        res=dict(p.map(T.run,[(b,aud[b]['z0'],100) for b in near]))
    json.dump(res,open('train_like_probe_near_100.json','w'),indent=1)
    for b,r in res.items():
        print('%-30s cut=%.3f gauss min=%.4f eps=%d | bang min=%.4f eps=%d'%(b,r['cut'],r['gauss']['min_z_ctrl'],r['gauss']['eps_below'],r['bang']['min_z_ctrl'],r['bang']['eps_below']))
