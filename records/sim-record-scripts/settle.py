import json, os
import mujoco, numpy as np
XML_DIR = "assets/unimals_100/xml"
def load(xml_path):
    s = open(xml_path).read().replace(
        '<!-- <geom name="floor" type="plane" pos="0 0 0" size="50 50 1" material="grid"/> -->',
        '<geom name="floor" type="plane" pos="0 0 0" size="50 50 1" contype="1" conaffinity="1" rgba="0.9 0.9 0.9 1"/>')
    return mujoco.MjModel.from_xml_string(s)
geo = {r['name']: r for r in json.load(open('geometry.json'))}
rows = []
for fn in sorted(os.listdir(XML_DIR)):
    n = fn[:-4]
    m = load(os.path.join(XML_DIR, fn)); d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2]); zmin = z0
    for _ in range(150 * 4):  # 3 s, zero action
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
        zmin = min(zmin, float(d.qpos[2]))
    rows.append((geo[n]['family'], n, z0, float(d.qpos[2]), zmin))
import collections
byz = collections.defaultdict(list)
for f, n, z0, zr, zmin in rows:
    byz[round(z0, 2)].append(zr)
print("z0 -> settled torso z (zero action, 3 s): min / median / max, n")
for z0 in sorted(byz):
    v = np.array(byz[z0])
    print(f"  z0={z0:.2f}: settled {v.min():.3f} / {np.median(v):.3f} / {v.max():.3f}  n={len(v)}  0.2z0={0.2*z0:.3f} 0.5z0={0.5*z0:.3f}")
inert = [r for r in rows if geo[r[1]]['inert_0.20']]
print("28 inert: settled torso z range %.3f..%.3f; strict cutoff 0.5*z0 vs settled: %s" % (
    min(r[3] for r in inert), max(r[3] for r in inert),
    sorted(set(round(0.5*r[2] - r[3], 3) for r in inert))[:6]))
n_strict_at_rest = sum(1 for r in rows if r[3] < 0.5 * r[2])
print("bodies whose zero-action resting torso is already below 0.5*z0 (Strict would fire at rest):", n_strict_at_rest)
print("  by family:", collections.Counter(r[0] for r in rows if r[3] < 0.5 * r[2]))
print("  of which inert-28:", sum(1 for r in rows if r[3] < 0.5 * r[2] and geo[r[1]]['inert_0.20']))
json.dump(rows, open('settle.json', 'w'))
