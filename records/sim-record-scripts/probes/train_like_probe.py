"""Training-like probe: can the torso reach tau*z0 (tau=0.10) under early-training actions?
CPU MuJoCo, same floor substitution as train_mjx.load_unimal_xml, training reset noise U[-0.01,0.01]
(qpos except x,y; qvel), FRAME_SKIP=4, z checked at control-step boundaries (as the trainer does)
and also at every physics substep. No termination (we want the full z trace)."""
import json, sys, numpy as np, mujoco
from multiprocessing import Pool
XML='assets/unimals_100/xml'
FRAME_SKIP=4
def load(b):
    s=open(f'{XML}/{b}.xml').read()
    s=s.replace('<!-- <geom name="floor" type="plane" pos="0 0 0" size="50 50 1" material="grid"/> -->',
                '<geom name="floor" type="plane" pos="0 0 0" size="50 50 1" contype="1" conaffinity="1" rgba="0.9 0.9 0.9 1"/>')
    assert 'name="floor"' in s
    return mujoco.MjModel.from_xml_string(s)
def episode(m,d,rng,mode,steps=1000,hold=5):
    mujoco.mj_resetData(m,d)
    init_q=d.qpos.copy(); init_v=d.qvel.copy()
    nq=rng.uniform(-0.01,0.01,m.nq); nq[:2]=0
    d.qpos[:]=init_q+nq; d.qvel[:]=init_v+rng.uniform(-0.01,0.01,m.nv)
    mujoco.mj_forward(m,d)
    zc=[]; zsub=np.inf
    a=np.zeros(m.nu)
    for t in range(steps):
        if mode=='gauss': a=np.clip(rng.normal(0,1,m.nu),-1,1)
        elif mode=='bang':
            if t%hold==0: a=rng.choice([-1.0,1.0],m.nu)
        d.ctrl[:]=a
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m,d); zsub=min(zsub,float(d.qpos[2]))
        zc.append(float(d.qpos[2]))
    return np.array(zc), zsub
def run(args):
    b,z0_aud,n_ep=args
    m=load(b); d=mujoco.MjData(m); mujoco.mj_resetData(m,d); z0=float(d.qpos[2])
    rng=np.random.default_rng(abs(hash(b))%2**32)
    out={'z0':z0,'z0_audit':z0_aud,'cut':0.1*z0}
    for mode in ['gauss','bang']:
        mins=[];subs=[];below_steps=0;eps_below=0;first=[]
        for e in range(n_ep):
            zc,zs=episode(m,d,rng,mode)
            mins.append(zc.min()); subs.append(zs)
            bl=np.where(zc<0.1*z0)[0]
            below_steps+=len(bl); eps_below+=int(len(bl)>0)
            if len(bl): first.append(int(bl[0]))
        out[mode]={'min_z_ctrl':float(min(mins)),'min_z_sub':float(min(subs)),'eps_below':eps_below,'n_ep':n_ep,
                   'steps_below':below_steps,'first_step_below':first[:5]}
    return b,out
if __name__=='__main__':
    aud=json.load(open('<scratch>'))['bodies']
    n_ep=int(sys.argv[1]); which=sys.argv[2]
    bodies=sorted(b for b in aud if aud[b]['geometric_inert']['0.10'])
    if which=='neg': bodies=[b for b in bodies if aud[b]['probe_min_torso_z']['min']<aud[b]['threshold']['0.10']]
    elif which=='one': bodies=bodies[:1]
    with Pool(int(sys.argv[3]) if len(sys.argv)>3 else 4) as p:
        res=dict(p.map(run,[(b,aud[b]['z0'],n_ep) for b in bodies]))
    json.dump(res,open(f'train_like_probe_{which}_{n_ep}.json','w'),indent=1)
    print('done',len(res))
