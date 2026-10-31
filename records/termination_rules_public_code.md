# Fall termination rules in public co-design code

Collected 2026-09-23. Every repository was cloned at the commit listed below, and all line numbers refer to that commit. Each link is a commit-pinned GitHub permalink, and each one was checked to return HTTP 200.

| Repo (official?) | Commit pinned | Evidence the repo is official |
|---|---|---|
| agrimgupta92/derl | `91e077e355b27bb74fa920c708d606ff9f4d611e` (HEAD, 2021-10-09) | README L5 links the Nature Communications paper |
| agrimgupta92/metamorph | `d49826475d4d22d8e63df694785833dc127c4e8d` (HEAD, 2022-03-22) | Same author and codebase lineage as DERL |
| Khrylx/Transform2Act | `23c5f0a502da035e077a4b555a6b71d4e202cbbc` (HEAD, 2023-12-06) | README L4-L7 ("ICLR 2022 (Oral)", OpenReview link) |
| Josh00-Lu/BodyGen | `de414ae974f9110e2ce00f0f0bf12a9965c0207a` (HEAD, 2025-10-21) | README L14 ("PyTorch implementation of BodyGen ... (ICLR 2025, Spotlight)"), OpenReview id cTR17xl89h |
| EvolutionGym/evogym | `f63c649c...` (HEAD, v2.x) and tag `1.0.0` = `9a1a5e7b26702184821e6e64587220ead2ab0e21` (the paper-era release) | README L6 cites the NeurIPS 2021 paper |

## Part 1: Termination-rule table

Notation: `z` is the world-frame height of the root or torso body, `ang` is the tilt angle, and `N` is the number of control steps. Any "absolute" threshold is a world-frame constant that does not scale with the morphology.

| Method | Termination condition (exact expression) | Reference height | Timeout handling | Alive bonus | Evidence links |
|---|---|---|---|---|---|
| **DERL** (Gupta et al. 2021), locomotion tasks FT and VT | `done = obs["torso_height"] <= fall_threshold`, with `torso_height = round(z_torso - terrain_z, 3)` (or `z_torso - 1` when the flat `Floor` module is used) and `fall_threshold = orig_height * STAND_HEIGHT_RATIO`, where `STAND_HEIGHT_RATIO = 0.5` (0.2 in the incline eval configs). Other terminations: terrain edge (`|y|` within 1 m of the arena side) whenever the `Terrain` module is used; wall contact and rotation only if they are enabled in `cfg.ENV.WRAPPERS`. | `orig_height = z_spawn - TERRAIN.SIZE[2] - BODY.FLOOR_OFFSET` = spawn torso height above the floor **minus 0.2 m**, which is the torso height at which the lowest body point touches the floor in the zero-joint-angle pose. The value is computed per morphology. See the DERL section below. | **Truncated, bootstrapped.** Gym `TimeLimit` (1000 steps) triggers `TimeLimitMask`, which sets `info["timeout"]`. In the GAE loop the timed-out transition gets `delta = 0` and target `ret = V(s)`, so earlier steps bootstrap through `V(s_last)`. The last step's reward is discarded. | **No.** `HEALTHY_REWARD = 0.0` is defined but never read. `StandReward` is applied, but `STAND_REWARD_WEIGHT = 0.0`. Reward = forward x-velocity minus control cost. | Threshold: [agent.py L15-L23](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/modules/agent.py#L15-L23), [agent.py L52-L53](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/modules/agent.py#L52-L53), [config.py L39-L40](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/config.py#L39-L40), [config.py L134-L136](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/config.py#L134-L136). Current height: [hfield.py L199-L224](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/wrappers/hfield.py#L199-L224). Check: [hfield.py L312-L329](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/wrappers/hfield.py#L312-L329). Wiring: [locomotion.py L78-L105](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/tasks/locomotion.py#L78-L105). Reward: [locomotion.py L27-L43](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/tasks/locomotion.py#L27-L43), [config.py L115-L123](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/config.py#L115-L123), [hfield.py L237-L250](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/wrappers/hfield.py#L237-L250). Timeout: [envs/\_\_init\_\_.py L3-L7](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/__init__.py#L3-L7), [ppo/envs.py L103-L110](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/algos/ppo/envs.py#L103-L110), [ppo.py L100-L110](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/algos/ppo/ppo.py#L100-L110), [buffer.py L45-L65](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/algos/ppo/buffer.py#L45-L65). Evo configs: [ft.yml L1-L19](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/configs/evo/ft.yml#L1-L19), [vt.yml L1-L17](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/configs/evo/vt.yml#L1-L17), [incline.yml L6](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/configs/eval/incline.yml#L6) |
| **MetaMorph** (Gupta et al. 2022) | Same as DERL: `obs["torso_height"] <= self.env.metadata["fall_threshold"]`, with `fall_threshold = orig_height * STAND_HEIGHT_RATIO`. The ratio defaults to 0.5, and **`configs/csr.yaml` sets it to 0.3**. `torso_height` is computed as in DERL. MetaMorph trains on DERL-generated UNIMAL XMLs, so the spawn height also comes from DERL's `set_head_pos` (+0.2 m). | `orig_height = z_spawn - TERRAIN.SIZE[2] - TERRAIN.FLOOR_OFFSET`, where `FLOOR_OFFSET = 0.2`. Unlike DERL, the offset is subtracted unconditionally (DERL skips it only for prebuilt `WALKER` agents). | **Truncated, bootstrapped.** The mechanism (`TimeLimitMask`, `timeout` mask, GAE) is identical to DERL. Limit: 1000 steps. | **No.** `HEALTHY_REWARD = 0.0` is unused. `STAND_REWARD_WEIGHT = 0.0`. `CTRL_COST_WEIGHT = 0.0`, so the reward is pure forward velocity. | [agent.py L17-L23](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/modules/agent.py#L17-L23), [agent.py L52-L53](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/modules/agent.py#L52-L53), [agent.py L312-L324](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/modules/agent.py#L312-L324), [hfield.py L200-L225](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/wrappers/hfield.py#L200-L225), [hfield.py L312-L328](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/wrappers/hfield.py#L312-L328), [config.py L39-L59](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/config.py#L39-L59), [config.py L138-L139](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/config.py#L138-L139), [csr.yaml L2-L4](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/configs/csr.yaml#L2-L4), [locomotion.py L29-L45](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/tasks/locomotion.py#L29-L45), [locomotion.py L82-L93](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/tasks/locomotion.py#L82-L93), [envs/\_\_init\_\_.py L3-L7](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/envs/__init__.py#L3-L7), [ppo/envs.py L159-L166](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/algos/ppo/envs.py#L159-L166), [ppo.py L91-L102](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/algos/ppo/ppo.py#L91-L102), [buffer.py L60-L80](https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/algos/ppo/buffer.py#L60-L80) |
| **Transform2Act** (Yuan et al. 2022) | For every env, `done = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)) and (self.control_nsteps < max_nsteps))`. **ant:** `height = s[2]` (root z); `min_height` 0.0 (default), `max_height` 2.0, `max_ang` **60 deg** (ant.yml); `ang` = tilt of the torso z-axis. **hopper:** `height = qpos[1]` (rootz slide, `ref=1.25`, so it equals world z); `min_height` **0.7** (default), `max_ang` **20 deg** (hopper.yml), where `ang = qpos[2]`. **gap:** `min_height` **1.0**, `max_ang` **20 deg** (gap.yml). **swimmer:** `done = not (control_nsteps < max_nsteps)` (time limit only). | **Fixed absolute constants** (ant 0.0, hopper 0.7, gap 1.0). They do not depend on the morphology. Spawn: ant root z is reset to 0.4 (`init_height` defaults to True); hopper spawns at z = 1.25 (`env_init_height` defaults to False); gap raises the body in 0.05 steps until it has no ground contact. | **Treated as terminal.** `max_nsteps` (1000) is folded into `done`, the agent stores `mask = 0 if done else 1`, and `estimate_advantages` uses `prev_value * masks[i]`, so the timeout step gets no bootstrap. | **hopper: yes, 1.0 per step** (default `alive_bonus` 1.0; hopper.yml does not override it). **gap: yes, 0.1** (gap.yml). **ant: no** (default 0.0). **swimmer: no.** | [ant.py L159-L178](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/envs/ant.py#L159-L178), [ant.py L282-L291](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/envs/ant.py#L282-L291), [hopper.py L157-L173](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/envs/hopper.py#L157-L173), [hopper.py L276-L300](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/envs/hopper.py#L276-L300), [gap.py L169-L188](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/envs/gap.py#L169-L188), [gap.py L290-L311](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/envs/gap.py#L290-L311), [swimmer.py L157-L168](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/envs/swimmer.py#L157-L168), [ant.yml L68-L69](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/cfg/ant.yml#L68-L69), [hopper.yml L67-L68](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/cfg/hopper.yml#L67-L68), [gap.yml L68-L82](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/cfg/gap.yml#L68-L82), [config.py L60-L69](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/utils/config.py#L60-L69), [hopper.xml L16-L20](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/assets/mujoco_envs/hopper.xml#L16-L20), [transform2act_agent.py L79-L81](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/agents/transform2act_agent.py#L79-L81), [transform2act_agent.py L228-L241](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/design_opt/agents/transform2act_agent.py#L228-L241), [common.py L5-L25](https://github.com/Khrylx/Transform2Act/blob/23c5f0a502da035e077a4b555a6b71d4e202cbbc/khrylib/rl/core/common.py#L5-L25) |
| **BodyGen** (Lu et al., ICLR 2025) | Transform2Act env code, with termination and truncation split: `termination = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))` and `truncation = not (self.control_nsteps < max_nsteps)`. Per task (`cfg -> env_name`, `done_condition`): **crawler** -> ant, `min_height` 0.0, `max_ang` 60 deg. **cheetah** -> hopper, `min_height` 0.7 (default), `max_ang` 20 deg. **glider-{regular,medium,hard}** and **walker-{regular,medium,hard}** -> walker, `min_height` 0.7 (default), `max_ang` 60 deg. **terraincrosser** -> gap, `min_height` 1.0, `max_ang` 20 deg. **swimmer**: `termination = False`. | **Fixed absolute constants** (0.0, 0.7, 1.0), as in Transform2Act. The ant resets root z to 0.4. Walker and hopper spawn at z = 1.25 (`env_init_height` defaults to False). | **Truncated, bootstrapped.** `delta = r + gamma * V(s') * (1 - next_termination) - V(s)`, with `V(s')` explicitly evaluated on `next_state` at every `done`. The GAE recursion is cut by `(1 - next_dones)`. Limit: 1000 control steps. | **No by default.** `alive_bonus` defaults to 0.0 in every env (changed from Transform2Act's 1.0 for the hopper), and no BodyGen cfg sets it. A global `reward_shift` (default 0.0) is added to every execution-stage reward and would act as an alive bonus if set. | [ant.py L166-L186](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/envs/ant.py#L166-L186), [ant.py L317-L326](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/envs/ant.py#L317-L326), [hopper.py L163-L180](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/envs/hopper.py#L163-L180), [walker.py L165-L182](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/envs/walker.py#L165-L182), [walker.py L312-L336](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/envs/walker.py#L312-L336), [gap.py L176-L194](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/envs/gap.py#L176-L194), [swimmer.py L164-L176](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/envs/swimmer.py#L164-L176), [crawler.yml L1-L14](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/cfg/crawler.yml#L1-L14), [cheetah.yml L1-L14](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/cfg/cheetah.yml#L1-L14), [walker-regular.yml L1-L14](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/cfg/walker-regular.yml#L1-L14), [glider-regular.yml L1-L14](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/cfg/glider-regular.yml#L1-L14), [terraincrosser.yml L1-L15](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/cfg/terraincrosser.yml#L1-L15), [walker.xml L26-L30](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/assets/mujoco_envs/walker.xml#L26-L30), [genesis_agent.py L179-L188](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/agents/genesis_agent.py#L179-L188), [genesis_agent.py L256-L289](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/agents/genesis_agent.py#L256-L289), [conf/config.yaml L30](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/design_opt/conf/config.yaml#L30), [Run_BodyGen.sh L3](https://github.com/Josh00-Lu/BodyGen/blob/de414ae974f9110e2ce00f0f0bf12a9965c0207a/scripts/Run_BodyGen.sh#L3) |
| **Evolution Gym** (Bhatia et al. 2021), walking tasks | **No fall or height rule** for Walker-v0, BridgeWalker-v0, or CaveCrawler-v0. `done` is set only when (a) the simulator reports instability (`reward -= 3.0`) or (b) the goal is reached (`com_x > 99*VOXEL_SIZE`, `+1.0`; 60 and 69 voxels for the other two tasks). Some traversal tasks do terminate on failure: UpStepper ends on a flip (`pi/2 - pi/12 < ort < 3pi/2 + pi/12`, -3), PlatformJumper ends when `com_y < 0.4` or on a flip (-3), and GapJumper ends when `com_y < 4*VOXEL_SIZE` (-3). | None for the walking tasks. The falling rules in PlatformJumper and GapJumper use absolute COM-y constants. | **Paper-era v1.0.0: treated as terminal.** It uses ikostrikov PPO with `--use-proper-time-limits` defaulting to False, so `bad_masks` are ignored. **Current HEAD (v2, SB3 PPO): truncated, bootstrapped.** SB3 adds `gamma * V(terminal_obs)` when `TimeLimit.truncated`. Limits: 500 steps (Walker, BridgeWalker) and 1000 (CaveCrawler). | **No.** Reward = change in COM x. | HEAD: [walk.py L39-L72](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/evogym/envs/walk.py#L39-L72), [base.py L43-L63](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/evogym/envs/base.py#L43-L63), [envs/\_\_init\_\_.py L16-L32](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/evogym/envs/__init__.py#L16-L32), [traverse.py L104-L111](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/evogym/envs/traverse.py#L104-L111), [traverse.py L427-L435](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/evogym/envs/traverse.py#L427-L435), [traverse.py L512-L516](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/evogym/envs/traverse.py#L512-L516), [examples/ppo/run.py L23-L62](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/examples/ppo/run.py#L23-L62). SB3 (tag v2.9.0): [on_policy_algorithm.py L234-L245](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/stable_baselines3/common/on_policy_algorithm.py#L234-L245), [dummy_vec_env.py L63-L66](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/stable_baselines3/common/vec_env/dummy_vec_env.py#L63-L66). v1.0.0: [walk.py L32-L65](https://github.com/EvolutionGym/evogym/blob/9a1a5e7b26702184821e6e64587220ead2ab0e21/evogym/envs/walk.py#L32-L65), [arguments.py L143-L147](https://github.com/EvolutionGym/evogym/blob/9a1a5e7b26702184821e6e64587220ead2ab0e21/examples/ppo/arguments.py#L143-L147), [envs.py L117-L124](https://github.com/EvolutionGym/evogym/blob/9a1a5e7b26702184821e6e64587220ead2ab0e21/examples/ppo/envs.py#L117-L124), [run.py L157-L163](https://github.com/EvolutionGym/evogym/blob/9a1a5e7b26702184821e6e64587220ead2ab0e21/examples/ppo/run.py#L157-L163), [run.py L186-L187](https://github.com/EvolutionGym/evogym/blob/9a1a5e7b26702184821e6e64587220ead2ab0e21/examples/ppo/run.py#L186-L187), and the ikostrikov submodule at `41332b78`: [storage.py L66-L105](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/41332b78dfb50321c29bade65f9d244387f68a60/a2c_ppo_acktr/storage.py#L66-L105) |

### Cross-method takeaways relevant to rankings

- Only DERL and MetaMorph set a **per-morphology** reference height. Transform2Act and BodyGen use **world-frame constants** (0.7 m or 1.0 m for planar bodies, 0.0 m plus a tilt limit for the ant). A short planar body that is otherwise stable can therefore end episodes simply because it is short.
- Transform2Act's hopper-family tasks pay an alive bonus of 1.0 per step and **treat the time limit as terminal**. As a result, a longer episode is directly worth more return. BodyGen removed the alive bonus (default 0.0) and added bootstrapping on truncation.
- DERL and MetaMorph pay no terminal penalty and no alive bonus. A fall costs a morphology only the forward reward it would have earned afterward. DERL selects on the episode sum of `__reward__forward` ([ft.yml L6-L7](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/configs/evo/ft.yml#L6-L7), [evo.py L46-L55](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/utils/evo.py#L46-L55)).

## Part 2: DERL's exact fall rule

### Equation

Let `z_t` be the world z of body `torso/0` after step `t`, `h(x_t, y_t)` the terrain height directly under the torso, `z_0` the torso's initial (spawn) height above the floor, and `c = BODY.FLOOR_OFFSET = 0.2 m`.

```
done_t  <=>  z_t - h(x_t, y_t)  <=  0.5 * (z_0 - 0.2)
```

where:

- `h = 1.0` m on flat ground, because both the floor box top and the hfield flat value equal `TERRAIN.SIZE[2] = 1`. The `Floor` module hard-codes `z - 1`. On hfield terrain `h` comes from the hfield lookup, clipped at 0 (except for incline tasks).
- `z_0 = z_spawn - TERRAIN.SIZE[2]`, and `z_spawn - TERRAIN.SIZE[2] = head_z = -min_i z_i^btm + 0.2`, where `z_i^btm` are the bottom-position sites of the torso and limbs in the zero-joint-angle pose, relative to the torso.
- The equivalent form is `z_t - h <= 0.5 * H_rest`, where `H_rest = -min_i z_i^btm = z_0 - 0.2` is the torso height at which the lowest body point just touches the floor.
- The comparison is `<=`, and `z_t - h` is rounded to 3 decimals before it.

### The two questions

1. **Where is 0.2 m subtracted? From the reference height only.** The 0.2 m clearance is added once to the spawn height (`set_head_pos`) and subtracted once when the reference is computed (`orig_height -= cfg.BODY.FLOOR_OFFSET`). The current height is `z_t - terrain_z` with no 0.2 m term. The 1.0 m terrain base (`TERRAIN.SIZE[2]`) is removed from both sides, which makes the two heights consistent "above ground" quantities. Only the clearance term is asymmetric.
2. **Cutoff for z_0 = 0.30 m.** If `z_0` is the initial (spawn) torso height above the floor, the cutoff is `0.5 * (0.30 - 0.20) = 0.05 m`. That is one third of the naive `0.5 * z_0 = 0.15 m`, and equals `0.167 * z_0`. If instead 0.30 m is the settled rest height `H_rest`, the body spawns at 0.50 m and the cutoff is 0.15 m. So "50% of original height" holds exactly only when "original" means the rest height with the 0.2 m clearance already removed.

The effective ratio of the cutoff to the spawn height depends on the morphology: `cutoff / z_0 = 0.5 * (1 - 0.2 / z_0)`.

| spawn height z_0 (m) | cutoff (m) | cutoff / z_0 |
|---|---|---|
| 0.25 | 0.025 | 0.10 |
| 0.30 | 0.050 | 0.17 |
| 0.40 | 0.100 | 0.25 |
| 0.50 | 0.150 | 0.30 |
| 1.00 | 0.400 | 0.40 |

Measured against spawn height, short bodies get a much more lenient fall rule. Measured against rest height `H_rest`, the ratio is a uniform 0.5.

### Code, quoted (DERL @ `91e077e`)

Spawn height gets the 0.2 m clearance. [derl/envs/morphology.py L270-L286](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/morphology.py#L270-L286):

```python
281        # Get z_axis of all the sites
282        z_coords = sim.data.site_xpos[btm_pos_site_ids][:, 2]
283        # All the z_axis are < 0. Select the smallest
284        btm_most_pos_idx = np.argmin(z_coords)
285        head_z = -1 * z_coords[btm_most_pos_idx] + cfg.BODY.FLOOR_OFFSET
286        self.unimal.set("pos", xu.arr2str([0, 0, round(head_z, 2)]))
```

The terrain base is added when the agent is merged into the world. [derl/envs/modules/agent.py L161-L164](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/modules/agent.py#L161-L164):

```python
161    # Update agent z pos based on starting terrain
162    pos = xu.str2arr(agent_body.get("pos"))
163    pos[2] += cfg.TERRAIN.SIZE[2]
164    agent_body.set("pos", xu.arr2str(pos))
```

The reference height and threshold subtract the terrain base and the 0.2 m clearance. [derl/envs/modules/agent.py L15-L23](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/modules/agent.py#L15-L23) and [L52-L53](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/modules/agent.py#L52-L53):

```python
15    def modify_xml_step(self, env, root, tree):
16        # Store agent height
17        worldbody = root.findall("./worldbody")[0]
18        head = xu.find_elem(worldbody, "body", "name", "torso/0")[0]
19        pos = xu.str2arr(head.get("pos"))
20        orig_height = pos[2] - cfg.TERRAIN.SIZE[2]
21        # Don't subtract offset for pre-built walkers
22        if not cfg.ENV.WALKER:
23            orig_height -= cfg.BODY.FLOOR_OFFSET
...
52        env.metadata["orig_height"] = round(orig_height, 2)
53        env.metadata["fall_threshold"] = orig_height * cfg.ENV.STAND_HEIGHT_RATIO
```

Constants. [derl/config.py L39-L40](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/config.py#L39-L40), [L134-L136](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/config.py#L134-L136), [L167-L168](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/config.py#L167-L168):

```python
39  # Vertical distance between the bottom most point of unimal and floor
40  _C.BODY.FLOOR_OFFSET = 0.2
...
134 # Healthy reward is 1 if head_pos >= STAND_HEIGHT_RATIO * head_pos in
135 # the xml i.e the original height of the unimal.
136 _C.ENV.STAND_HEIGHT_RATIO = 0.5
...
167 # Size of the "floor/0" x, y, z
168 _C.TERRAIN.SIZE = [25, 20, 1]
```

(The comment at L134-L135 says "head_pos in the xml", but the code applies the ratio to the XML head position minus 0.2 m.)

The current height has no 0.2 m term. [derl/envs/wrappers/hfield.py L204-L224](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/wrappers/hfield.py#L204-L224):

```python
204    def observation(self, obs):
205        x_pos, y_pos, z_pos = self.sim.data.get_body_xpos("torso/0")
206
207        # Height of ground is constant if Floor
208        if "Floor" in cfg.ENV.MODULES:
209            obs["torso_height"] = round(z_pos - 1, 3)
210            return obs
211
212        row_idx, col_idx = obs["hfield_idx"]
213        try:
214            terrain_z = self.metadata["hfield"][row_idx, col_idx]
215            # In case of gap terrain_z will be negative, clip at 0
216            if cfg.ENV.TASK not in ["incline", "push_box_incline"]:
217                terrain_z = max(0, terrain_z)
...
223        obs["torso_height"] = round(z_pos - terrain_z, 3)
```

The check itself. [derl/envs/wrappers/hfield.py L312-L329](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/wrappers/hfield.py#L312-L329):

```python
312 class TerminateOnFalling(gym.Wrapper):
313     """Teriminate episode if torso falls below a certain height."""
314
315     def __init__(self, env):
316         super().__init__(env)
317         self.fall_threshold = env.metadata["fall_threshold"]
318
319     def step(self, action):
320         obs, rew, done, info = self.env.step(action)
321         if self.has_fallen(obs):
322             done = True
323         return obs, rew, done, info
324
325     def has_fallen(self, obs):
326         if obs["torso_height"] <= self.fall_threshold:
327             return True
```

The flat hfield value equals `TERRAIN.SIZE[2]`, so `terrain_z = 1` on flat ground in the FT and VT evolution runs (which use `MODULES: ["Agent", "Terrain"]`). See [terrain.py L235-L266](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/modules/terrain.py#L235-L266) and [ft.yml L9](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/configs/evo/ft.yml#L9).

### What the DERL paper says (for contrast)

arXiv:2102.02202v1, Methods, "Rewards" paragraph: "In addition, we use early termination across all environments when we detect a fall. We consider an agent to have fallen if the head of the agents falls below 50% of it's original height. We found employing this early termination criterion was essential in ensuring diverse gaits. Without early termination, almost all agents would immediately fall and move in a snake like gait." The "Stability" paragraph adds: "Note that we use the violation of this same condition for early termination of the episode." The paper does not mention the 0.2 m clearance. This quote is from arXiv v1, not the Nature Communications typeset version.

## Search notes

- **BodyGen repository:** found by a GitHub repository search. `Josh00-Lu/BodyGen` is the only match with a README claiming the ICLR 2025 implementation. `bastian-knierim/BodyGenExtend` is a third-party extension.
- **MetaMorph source files:** the repo contains no `morphology.py`. Its UNIMAL XMLs are downloaded (README L33-L36), so their spawn heights come from DERL's generator. The downloaded XMLs were not opened.
- **DERL paper text:** from arXiv v1, not the Nature Communications version of the Methods text.
