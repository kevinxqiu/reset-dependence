#!/usr/bin/env python3
"""Write croissant.json, the Croissant 1.1 metadata of the run-summary dataset.

The file is written next to this script. The sha256 and contentSize of
runs.zip, logs.zip and panels.zip, and the sha256 of the CSV tables inside
them, are filled in from the archives when they are present in --archives
(the directory that holds the built dataset); otherwise the values already
in croissant.json are kept.

Validate with:
    mlcroissant validate --jsonld release/dataset/croissant.json
    python release/dataset/load_croissant_records.py --archives <built dir>

The file holds the core fields of Croissant 1.1 and the minimal Responsible
AI (RAI) items that the NeurIPS 2026 Evaluations and Datasets track asks for
(https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting and
https://blog.neurips.cc/2026/05/04/responsible-ai-metadata-requirements-for-the-evaluations-and-datasets-track-neurips-2026/):
data limitations, data biases, personal or sensitive information, data use
cases, social impact (rai:*), synthetic data (rai:hasSyntheticData), source
datasets (prov:wasDerivedFrom) and provenance activities
(prov:wasGeneratedBy). REQUIRED_RAI below lists them, and main() refuses to
write a file that misses one.

The archives are hosted as assets of release v1.1.2 of the code repository
(DATASET_URL and CONTENT_BASE below). There is no separate dataset record or DOI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The archives are assets of this GitHub release of the code repository.
DATASET_URL = "https://github.com/kevinxqiu/reset-dependence/releases/tag/v1.1.2"
CONTENT_BASE = "https://github.com/kevinxqiu/reset-dependence/releases/download/v1.1.2"
CODE_URL = "https://github.com/kevinxqiu/reset-dependence"
VERSION = "1.1.2"

# JSON-LD context of Croissant 1.1 (spec appendix 1) plus the PROV-O prefix.
CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "annotation": "cr:annotation",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "excludes": "cr:excludes",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "prov": "http://www.w3.org/ns/prov#",
    "readLines": "cr:readLines",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "sdVersion": "cr:sdVersion",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "unArchive": "cr:unArchive",
    "value": "cr:value",
}

# Minimal RAI items of the NeurIPS 2026 E&D track (see the module docstring).
REQUIRED_RAI = (
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
    "prov:wasDerivedFrom",
    "prov:wasGeneratedBy",
)
# Core fields that Croissant 1.1 requires (spec section "Required").
REQUIRED_CORE = ("@context", "@type", "conformsTo", "description", "license", "name", "url",
                 "creator", "datePublished", "distribution")

DESCRIPTION = (
    "Run summaries behind the NeurIPS 2026 Evaluations and Datasets paper 'Reset Dependence Is "
    "Embodied: Training Protocol Changes Which Morphologies Appear Optimal'. The 100 training bodies of MetaMorph's "
    "UNIMAL-100 benchmark, which MetaMorph chose from the UNIMAL bodies that DERL evolved, were trained "
    "with PPO for 5M environment steps under fall rules that end a training "
    "episode when the torso drops below a fraction tau of its reset height (Reset-free tau = 0, "
    "Standard tau = 0.20, Strict tau = 0.50, plus 0.10 and 0.35), with seeds 42, 43 and 44, and "
    "each controller was scored over 20 episodes by three evaluators. The dataset holds "
    "one JSON summary per run for the 900-run corpus and the other UNIMAL panels (3,306 runs "
    "in all, including SAC, retraining, time-limit, contact-termination, non-evolved-body, "
    "longer-budget and existing-robot panels), a flat CSV of all runs, structural features of the "
    "bodies, the zero-action and probe simulation records, joint-angle traces of nine illustrative "
    "controllers, and training logs where they were kept. No policy checkpoints are included. The UNIMAL "
    "body files are not included; the code repository downloads them from the upstream release and "
    "checks their hashes. With the CPU scripts in the code repository these files recompute the "
    "statistics and figures that the paper reports for the UNIMAL panels. The numbers script recomputes 2,401 "
    "values, and the end of its report (records/camera_ready_numbers.md) lists the printed numbers it does not "
    "recompute with their sources: mostly settings, dates, decision rules and quoted values, plus results kept "
    "outside the run summaries such as the cross-validated structure fits. "
    "panels.zip holds the five control panels trained during the review discussion "
    "outside the UNIMAL corpus (5,860 cells): Humanoid fall rules, Humanoid time-limit handling, Ant reset "
    "initialization, Ant protocol breadth (reset initialization, time-limit handling, observation noise) and "
    "Evolution Gym time-limit handling. It holds one CSV row per morphology, protocol, seed and hardware class "
    "with the score each frozen report ranks, the Humanoid evaluation curves, and the frozen reports, panel "
    "specifications, contracts and inventories; scripts/review_panels.py in the code repository recomputes "
    "every statistic and cell count in those reports from the tables."
)

PERSON = [
    {"@type": "sc:Person", "name": "Kevin Qiu", "affiliation": {"@type": "sc:Organization", "name": "University of Warsaw"}},
    {"@type": "sc:Person", "name": "Zhao Yang", "affiliation": {"@type": "sc:Organization", "name": "Vrije Universiteit Amsterdam"}},
    {"@type": "sc:Person", "name": "Marek Cygan", "affiliation": {"@type": "sc:Organization", "name": "University of Warsaw"}},
]

CITE = (
    "@inproceedings{qiu2026reset,\n"
    "  title     = {Reset Dependence Is Embodied: Training Protocol Changes Which Morphologies Appear Optimal},\n"
    "  author    = {Qiu, Kevin and Yang, Zhao and Cygan, Marek},\n"
    "  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},\n"
    "  year      = {2026}\n"
    "}"
)

EVALS = (
    ("common", "eval_tau020", "common evaluator (an episode also ends when the torso falls below 0.20 of its reset height)"),
    ("neutral", "eval_neutral", "neutral evaluator (no fall clause; episodes run 1,000 steps unless the torso leaves the upper bound)"),
    ("strict", "eval_tau050", "Strict evaluator (an episode also ends when the torso falls below 0.50 of its reset height)"),
)
METRICS = (
    ("per_step_reward", "sc:Float", "Mean over 20 episodes of the episode return divided by the episode length (PSR), the paper's main score"),
    ("per_step_reward_std", "sc:Float", "Standard deviation of PSR over the 20 episodes"),
    ("fitness", "sc:Float", "Mean episode return over the 20 episodes (reward v_x - 0.001 |a|^2 per control step)"),
    ("fitness_std", "sc:Float", "Standard deviation of the episode return"),
    ("distance", "sc:Float", "Mean final x position of the torso in meters (the reset x position is 0)"),
    ("distance_std", "sc:Float", "Standard deviation of the final x position"),
    ("mean_ep_len", "sc:Float", "Mean episode length in control steps (at most 1,000)"),
    ("ep_len_std", "sc:Float", "Standard deviation of the episode length"),
    ("fall_rate", "sc:Float", "Fraction of the 20 episodes that ended before 1,000 steps (fall clause or upper bound)"),
)


def field(rs: str, name: str, dtype: str, desc: str, fileset: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"{rs}/{name}",
        "name": name,
        "description": desc,
        "dataType": dtype,
        "source": {"fileObject": {"@id": fileset}, "extract": {"column": name}},
    }


def runs_fields() -> list[dict]:
    fs = "runs-table"
    out = [
        field("runs", "panel", "sc:Text", "Panel id: corpus, intermediate_tau, sac, retraining, no_time_limit, time_limit_duration, rule_by_time_limit, contact_termination, non_evolved, longer_budget_5m or existing_robots (see the data card)", fs),
        field("runs", "results_file", "sc:Text", "Path inside runs.zip of the JSON run summary this row comes from", fs),
        field("runs", "body", "sc:Text", "Body name: a UNIMAL body (floor-*, vt-*, mvt-*), a non-evolved body (gen-*) or an existing robot", fs),
        field("runs", "family", "sc:Text", "floor, vt or mvt (environment DERL evolved the body in), non_evolved, or robot", fs),
        field("runs", "algorithm", "sc:Text", "PPO or SAC", fs),
        field("runs", "backend", "sc:Text", "Trainer implementation label written by the training script", fs),
        field("runs", "tau", "sc:Float", "Training fall threshold as a fraction of the reset height (0 means no fall clause)", fs),
        field("runs", "seed", "sc:Integer", "Training seed", fs),
        field("runs", "timesteps", "sc:Integer", "Training budget in environment steps", fs),
        field("runs", "termination_mode", "sc:Text", "height or contact-duration; empty where the run record does not store it (height-based rule)", fs),
        field("runs", "contact_termination_steps", "sc:Integer", "For contact termination: consecutive torso-ground contact steps that end an episode; empty otherwise", fs),
        field("runs", "max_steps", "sc:Integer", "Training time limit in steps when it differs from 1,000 (5000001 means no time limit within the budget); empty means 1,000", fs),
    ]
    for tag, key, edesc in EVALS:
        for m, dtype, mdesc in METRICS:
            out.append(field("runs", f"{tag}_{m}", dtype, f"{mdesc}; {edesc}; from {key}.{m} in the JSON summary", fs))
    out += [
        field("runs", "train_time_s", "sc:Float", "Recorded wall time of training plus evaluation in seconds", fs),
        field("runs", "git_commit", "sc:Text", "Code commit recorded by the run (later panels only; empty for the corpus, intermediate-tau and SAC runs)", fs),
        field("runs", "slurm_job_id", "sc:Integer", "SLURM job id recorded by the run (later panels only)", fs),
        field("runs", "host", "sc:Text", "Pseudonymous GPU node id (gpu-node-NN; later panels only)", fs),
        field("runs", "device_kind", "sc:Text", "GPU model recorded by the run (later panels only)", fs),
    ]
    return out


FEATURES = (
    ("body", "sc:Text", "Body name"),
    ("body_set", "sc:Text", "unimal_100 (the evolved corpus) or non_evolved_60"),
    ("family", "sc:Text", "floor, vt, mvt or non_evolved"),
    ("n_bodies", "sc:Integer", "Number of rigid bodies, torso included"),
    ("n_joints", "sc:Integer", "Number of joints"),
    ("n_motors", "sc:Integer", "Number of actuators"),
    ("n_geoms", "sc:Integer", "Number of collision geoms"),
    ("init_height", "sc:Float", "Reset height z0 of the torso in meters"),
    ("tree_depth", "sc:Integer", "Depth of the kinematic tree: links on the longest chain below the torso"),
    ("total_volume", "sc:Float", "Summed volume of the sphere and capsule geoms in cubic meters (a mass proxy)"),
    ("n_first_level_limbs", "sc:Integer", "Number of limbs attached directly to the torso"),
)


# Review-period panels (panels.zip): one cells.csv per panel with the same columns, and the
# evaluation curves of the two Humanoid panels. (folder, record set, description, has progress.csv)
PANEL_TABLES = (
    ("humanoid-fall-rule", "humanoid_fall_rule",
     "Humanoid fall rules (panel humanoid-primary-gh200-20260725): 3 pools of 30 parametric Humanoid bodies, "
     "training without a fall rule, with the native height range, or with cutoffs at 0.2 and 0.5 of the nominal "
     "root height, 5 seeds, Brax PPO in MJX on GH200 GPUs (1,800 cells).", True),
    ("humanoid-time-limit", "humanoid_time_limit",
     "Terminal arm of the Humanoid time-limit panel (panel humanoid-timeout-factorial-gh200-20260727): the same "
     "90 bodies and 5 seeds with the time limit treated as terminal, under the native fall rule and without a "
     "fall rule (900 cells). Its bootstrapped arm is the native and none cells of humanoid_fall_rule_cells.", True),
    ("ant-reset-init", "ant_reset_init",
     "Ant reset initialization (panel ant-reset-init-h100-20260725): 20 parametric Ant bodies, initial states "
     "sampled with reset noise (native) against one nominal initial state (reset-deterministic), 4 seeds, "
     "Brax PPO in MJX on H100 GPUs (160 cells).", False),
    ("ant-breadth", "ant_breadth",
     "Ant protocol breadth (panel ant-protocol-breadth-gh200-20260725): 3 pools of 30 parametric Ant bodies, "
     "the native protocol against deterministic resets, terminal time limits or observation noise, 5 seeds, "
     "GH200 GPUs (1,800 cells). Scores and file hashes were recovered from the raw records on 2026-09-24; the "
     "evaluation detail columns are empty.", False),
    ("evogym-time-limit", "evogym_time_limit",
     "Evolution Gym time-limit handling (panel evogym-wave1-cpu-20260724): 3 pools of 20 voxel bodies on "
     "Walker-v0 and BridgeWalker-v0, the 500-step time limit bootstrapped against terminal, 5 seeds, "
     "Stable-Baselines3 PPO on CPU (1,200 cells).", False),
)
PANEL_CELL_FIELDS = (
    ("panel_id", "sc:Text", "Frozen panel id (humanoid-primary-gh200-20260725, humanoid-timeout-factorial-gh200-20260727, ant-reset-init-h100-20260725, ant-protocol-breadth-gh200-20260725 or evogym-wave1-cpu-20260724)"),
    ("space", "sc:Text", "Design space: humanoid, ant or evogym"),
    ("pool_id", "sc:Integer", "Independently generated pool of 100 bodies (0, 1 or 2); rankings are compared within a pool"),
    ("task", "sc:Text", "Evolution Gym task (Walker-v0 or BridgeWalker-v0); empty for Humanoid and Ant"),
    ("morphology_index", "sc:Integer", "Index of the body in its pool"),
    ("morphology_id", "sc:Text", "Body name recorded by the run (for example humanoid-p0-003); empty for Evolution Gym"),
    ("body_sha256", "sc:Text", "SHA-256 of the body definition recorded by the run (generated MuJoCo XML or Evolution Gym voxel body); empty for the Ant breadth panel"),
    ("protocol", "sc:Text", "Training protocol of the cell, as named in the panel specification and report"),
    ("seed", "sc:Integer", "Training seed"),
    ("hardware", "sc:Text", "Hardware class of the run: gh200 or h100 (GPU), or cpu"),
    ("score", "sc:Float", "The score the panel's frozen report ranks. Humanoid and Ant: mean return of the final evaluation over 128 episodes of up to 1,000 steps with the common evaluator (no fall termination, no healthy reward), Brax eval/episode_reward. Evolution Gym: mean return over 20 evaluation episodes of up to 500 steps (aggregate_mean_return)"),
    ("termination", "sc:Text", "Fall rule during training: none, native (root height below the lower bound of the Gymnasium healthy range, 0.2 m for Ant and 1.0 m for Humanoid), relative-0.2 or relative-0.5 (root height below that fraction of the body's nominal root height); empty for Evolution Gym, which has no fall rule"),
    ("terminal_timeout", "sc:Boolean", "True if reaching the training time limit ended the episode as terminal (no bootstrapping); false if the value was bootstrapped"),
    ("deterministic_reset", "sc:Boolean", "True if every training episode started from the nominal initial state; false if noise was added to the initial positions and velocities; empty for Evolution Gym"),
    ("observation_noise_std", "sc:Float", "Standard deviation of Gaussian noise added to normalized observations during training; empty for Evolution Gym"),
    ("episode_length", "sc:Integer", "Training time limit in steps (1,000 for Humanoid and Ant, 500 for Evolution Gym)"),
    ("requested_steps", "sc:Integer", "Requested training budget in environment steps"),
    ("actual_steps", "sc:Integer", "Environment steps actually trained (the trainer rounds the budget up to whole iterations)"),
    ("eval_reward_std", "sc:Float", "Standard deviation of the evaluation return (Brax eval/episode_reward_std; for Evolution Gym over the 20 episodes); empty for the Ant breadth panel"),
    ("eval_episode_length", "sc:Float", "Mean evaluation episode length in steps; empty for the Ant breadth panel"),
    ("eval_forward_reward", "sc:Float", "Forward-progress part of the evaluation return (Brax eval/episode_forward_reward); Humanoid and Ant only, empty for the Ant breadth panel"),
    ("eval_control_cost", "sc:Float", "Control-cost part of the evaluation return (Brax eval/episode_control_cost; score = forward reward minus control cost); Humanoid and Ant only, empty for the Ant breadth panel"),
    ("train_seconds", "sc:Float", "Training wall time in seconds (Brax training/walltime; Evolution Gym training_elapsed_seconds); empty for the Ant breadth panel"),
    ("git_revision", "sc:Text", "Code revision recorded by the run (a commit of the authors' development repository)"),
    ("slurm_job_id", "sc:Integer", "SLURM job id recorded by the run; empty for the Ant breadth panel"),
    ("result_sha256", "sc:Text", "SHA-256 of the run's raw result file (result.json; evaluation.json for Evolution Gym), which is not distributed"),
    ("config_sha256", "sc:Text", "SHA-256 of the run's raw config.json, which is not distributed because it holds storage paths and a root binding token"),
)
PANEL_PROGRESS_FIELDS = (
    ("panel_id", "sc:Text", "Frozen panel id"),
    ("pool_id", "sc:Integer", "Pool of the body"),
    ("morphology_index", "sc:Integer", "Index of the body in its pool"),
    ("protocol", "sc:Text", "Training protocol of the cell"),
    ("seed", "sc:Integer", "Training seed"),
    ("hardware", "sc:Text", "Hardware class of the run"),
    ("eval_index", "sc:Integer", "Evaluation point, 0 to 5 (0 is before training)"),
    ("step", "sc:Integer", "Environment steps trained at this evaluation point"),
    ("eval_episode_reward", "sc:Float", "Mean evaluation return at this point, with the same evaluator as the score; the last point equals the cell's score"),
)

# Source datasets (prov:wasDerivedFrom). None of their files is redistributed.
SOURCES = [
    {
        "@id": "https://github.com/agrimgupta92/metamorph",
        "@type": "prov:Entity",
        "name": "UNIMAL-100 benchmark of MetaMorph (training set chosen from the bodies evolved by DERL)",
        "description": "The 100 UNIMAL training bodies of the UNIMAL-100 benchmark (unimals_100/train/xml) that every UNIMAL panel trains on (Gupta, Fan, Ganguli and Fei-Fei, ICLR 2022). MetaMorph chose them from the bodies that DERL evolved (Gupta, Savarese, Ganguli and Fei-Fei, Nature Communications 12, 5721, 2021). The upstream repository declares no license, so the files are not included. The code repository downloads them and checks their SHA-256.",
        "url": "https://github.com/agrimgupta92/metamorph",
    },
    {
        "@id": "https://github.com/agrimgupta92/derl",
        "@type": "prov:Entity",
        "name": "DERL code base (UNIMAL generator)",
        "description": "Generator used to sample the 60 non-evolved bodies (gen-*) of the non_evolved panel. The repository declares no license; the generated XMLs are not included, only their features and run summaries.",
        "url": "https://github.com/agrimgupta92/derl",
    },
    {
        "@id": "https://github.com/google-deepmind/mujoco_menagerie",
        "@type": "prov:Entity",
        "name": "MuJoCo Menagerie robot models",
        "description": "MJX models of Unitree Go2, ANYbotics ANYmal C, ToddlerBot 2XC and Unitree G1 used by the existing_robots panel. Not included; each model keeps its own license in the Menagerie repository.",
        "url": "https://github.com/google-deepmind/mujoco_menagerie",
    },
]
UNIMAL_SOURCES = list(SOURCES)
# Sources of the review-period panels (panels.zip).
PANEL_SOURCES = [
    {
        "@id": "https://github.com/Farama-Foundation/Gymnasium",
        "@type": "prov:Entity",
        "name": "Gymnasium MuJoCo Ant and Humanoid models",
        "description": "The parametric Ant and Humanoid bodies of the review-period panels were generated by scaling the limbs of Gymnasium's Ant and Humanoid MuJoCo models. The generated body files are not included; the tables record the SHA-256 of each.",
        "url": "https://github.com/Farama-Foundation/Gymnasium",
    },
    {
        "@id": "https://github.com/EvolutionGym/evogym",
        "@type": "prov:Entity",
        "name": "Evolution Gym",
        "description": "Walker-v0 and BridgeWalker-v0 tasks and the voxel bodies of the Evolution Gym panel (Bhatia et al., NeurIPS 2021). The body definitions are not included; the tables record the SHA-256 of each.",
        "url": "https://github.com/EvolutionGym/evogym",
    },
]
SOURCES = UNIMAL_SOURCES + PANEL_SOURCES

AUTHORS_AGENT = [{"@type": "prov:Agent", "name": p["name"]} for p in PERSON]

# Provenance activities (prov:wasGeneratedBy).
ACTIVITIES = [
    {
        "@type": "prov:Activity",
        "name": "Controller training and evaluation",
        "description": "PPO (and, for one panel, SAC) training of one controller per body, rule and seed in MuJoCo MJX on GPUs of a SLURM cluster, followed by CPU MuJoCo evaluation over 20 episodes with three evaluators (PPO actions sampled, SAC actions deterministic). Each run wrote one JSON summary. The 900-run corpus was trained in several launches from 5 to 8 April 2026, and the other panels of runs.zip later in April 2026. Trainers: corpus_revision/ and scripts/ in the code repository.",
        "prov:startedAtTime": "2026-04-05",
        "prov:used": [{"@id": s["@id"]} for s in UNIMAL_SOURCES],
        "prov:wasAssociatedWith": AUTHORS_AGENT,
    },
    {
        "@type": "prov:Activity",
        "name": "Review-period control panels",
        "description": "Training of the Humanoid, Ant and Evolution Gym panels during the review discussion, between 25 July and 1 August 2026: Brax PPO in MuJoCo MJX on GH200 and H100 GPUs for Humanoid and Ant, Stable-Baselines3 PPO on CPU for Evolution Gym. Each panel's cells, comparisons and analysis settings were fixed in a specification and contract before its analysis was run, and a frozen analysis wrote one report per panel (the Humanoid time-limit report is a composite of two panels). Scores and file hashes of the Ant breadth panel were recovered from its raw records on 2026-09-24.",
        "prov:startedAtTime": "2026-07-25",
        "prov:endedAtTime": "2026-08-01",
        "prov:used": [{"@id": s["@id"]} for s in PANEL_SOURCES],
        "prov:wasAssociatedWith": AUTHORS_AGENT,
    },
    {
        "@type": "prov:Activity",
        "name": "Simulation probe records",
        "description": "CPU MuJoCo zero-action rollouts, settle heights, random-action searches and training-like torso probes of the UNIMAL bodies (sim-records/), made with records/sim-record-scripts/ in the code repository.",
        "prov:startedAtTime": "2026-10-05",
        "prov:endedAtTime": "2026-10-05",
        "prov:used": [{"@id": SOURCES[0]["@id"]}],
        "prov:wasAssociatedWith": AUTHORS_AGENT,
    },
    {
        "@type": "prov:Activity",
        "name": "Packaging and masking",
        "description": "GPU host names replaced by stable pseudonyms (gpu-node-NN), home directories and the UNIMAL XML directory masked in recorded paths and logs, empty lock files dropped, the JSON summaries and morphology features flattened into tables/runs.csv and tables/morph_features.csv, and the archives written with fixed timestamps so that the build is byte-reproducible. Apart from this masking, the run summaries are unchanged. For panels.zip, a fixed list of fields was copied from each run's raw records into the CSV tables (no host names or paths), every score was checked against the frozen report that used it, root binding tokens and storage locations were removed from the panel specifications as the frozen reports do, and one internal design-note path was masked in two specifications, the composite specification and the Evolution Gym report.",
        "prov:wasAssociatedWith": AUTHORS_AGENT,
    },
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(archives: Path | None) -> dict:
    prev = {}
    old = HERE / "croissant.json"
    if old.is_file():
        for d in json.loads(old.read_text()).get("distribution", []):
            if d.get("@type") == "cr:FileObject":
                prev[d["name"]] = (d.get("sha256"), d.get("contentSize"))

    def file_object(name: str, desc: str) -> dict:
        digest, size = prev.get(name, (None, None))
        if archives and (archives / name).is_file():
            digest, size = sha256(archives / name), f"{(archives / name).stat().st_size} B"
        if digest is None:
            raise SystemExit(f"ERROR: no sha256 for {name}; pass --archives with the built dataset.")
        return {
            "@type": "cr:FileObject",
            "@id": name,
            "name": name,
            "description": desc,
            "contentUrl": f"{CONTENT_BASE}/{name}",
            "encodingFormat": "application/zip",
            "contentSize": size,
            "sha256": digest,
        }

    def member_object(id_: str, desc: str, container: str, member: str) -> dict:
        digest = prev.get(id_, (None, None))[0]
        if archives and (archives / container).is_file():
            import zipfile

            with zipfile.ZipFile(archives / container) as zf:
                digest = hashlib.sha256(zf.read(member)).hexdigest()
        if digest is None:
            raise SystemExit(f"ERROR: no sha256 for {id_}; pass --archives with the built dataset.")
        return {
            "@type": "cr:FileObject",
            "@id": id_,
            "name": id_,
            "description": desc,
            "containedIn": {"@id": container},
            "contentUrl": member,
            "encodingFormat": "text/csv",
            "sha256": digest,
        }

    def file_set(id_: str, desc: str, container: str, fmt: str, includes: str) -> dict:
        return {
            "@type": "cr:FileSet",
            "@id": id_,
            "name": id_,
            "description": desc,
            "containedIn": {"@id": container},
            "encodingFormat": fmt,
            "includes": includes,
        }

    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "name": "reset-dependence-runs",
        "alternateName": "Reset Dependence Is Embodied: run summaries",
        "description": DESCRIPTION,
        "conformsTo": ["http://mlcommons.org/croissant/1.1", "http://mlcommons.org/croissant/RAI/1.0"],
        "citeAs": CITE,
        "creator": PERSON,
        "datePublished": "2026-11-06",
        "dateCreated": "2026-04-05",
        "dateModified": "2026-10-13",
        "version": VERSION,
        "inLanguage": "en",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "sdLicense": "https://creativecommons.org/licenses/by/4.0/",
        "url": DATASET_URL,
        "sameAs": CODE_URL,
        "keywords": ["robot co-design", "morphology", "reinforcement learning", "evaluation protocol",
                     "termination", "UNIMAL", "MuJoCo", "benchmark audit"],
        "isLiveDataset": False,
        "rai:dataCollection": "All records were produced by the authors in simulation. The run summaries in runs.zip come from GPU PPO and SAC training in MuJoCo MJX in April 2026, and the zero-action and probe records were produced on CPU on 2026-10-05. The review-period panels in panels.zip were trained between 25 July and 1 August 2026 (Brax PPO in MuJoCo MJX for Humanoid and Ant, Stable-Baselines3 PPO in Evolution Gym). No human subjects or personal data are involved.",
        "rai:dataCollectionType": "Simulation",
        "rai:dataPreprocessingProtocol": "Run summaries are the JSON files written by the training scripts, unchanged except that GPU host names are replaced by stable pseudonyms (gpu-node-NN) and home directories in recorded paths are masked. Job ids and code commits are kept. The CSV tables are flattened copies of the JSON summaries and morphology features. The review-period panel tables (panels.zip) copy a fixed list of fields from each run's raw records, with no host names or paths; every score equals the one the panel's frozen report used. The frozen reports are copied unchanged except for one masked design-note path in the Evolution Gym report, and the panel specifications are shipped without their root binding tokens and storage locations, as in the reports.",
        "rai:dataLimitations": "Two defects in the fall thresholds (described in the paper): the Standard cutoff lies below the torso radius for 28 bodies, and the Strict cutoff ends episodes during the reset drop for ten VT bodies (an eleventh body, from the Floor family, is cut off without an identified mechanism). The corpus, intermediate-tau and SAC records do not store the code version, full settings or GPU. No policy checkpoints were saved. Results hold for one task (forward locomotion), one PPO preset and three seeds. The review-period panels are small (20 to 30 bodies per pool, four or five seeds), use other trainers, budgets and evaluators than the UNIMAL panels, and the two arms of the Humanoid time-limit panel were trained with different code revisions.",
        "rai:dataBiases": "The 100 bodies are MetaMorph's selection from the bodies that DERL evolved and are not a random sample of designs. Family sizes are unequal (41 Floor, 37 VT, 22 MVT).",
        "rai:dataUseCases": "Recomputing the paper's statistics; studying how training-protocol choices change morphology rankings; benchmarking reporting practices in co-design. Not suitable for training or deploying controllers on hardware.",
        "rai:dataSocialImpact": "Low. The data describe simulated locomotion and contain no personal information.",
        "rai:personalSensitiveInformation": "None.",
        # Not a term of the RAI 1.0 vocabulary (mlcroissant ignores it); NeurIPS 2026 lists it among the minimal items.
        "rai:hasSyntheticData": True,
        "rai:dataCollectionMissingData": "No policy checkpoints were saved. No training logs exist for the corpus and SAC runs. The corpus, intermediate-tau and SAC summaries do not record the code commit, full settings or GPU. For the review-period panels, the raw run records, policy checkpoints and generated body files are not included (the tables keep the SHA-256 of each raw result and configuration file and of each body); the Ant breadth panel's report.json was not kept, so its frozen summary table is shipped instead, and its table lacks the evaluation details, body hashes, wall times and job ids. The A2C panel records are not included.",
        "rai:dataReleaseMaintenancePlan": "Versioned GitHub releases of the code repository; corrections are published as new releases with a changelog in the data card.",
        "prov:wasDerivedFrom": SOURCES,
        "prov:wasGeneratedBy": ACTIVITIES,
        "distribution": [
            file_object("runs.zip", "JSON run summaries (results/), flat CSV tables (tables/), simulation probe records (sim-records/) and gait traces (gait-traces/)."),
            file_object("logs.zip", "Training logs where they were kept (results/<panel>/logs/ and per-run logs/), host names and home paths masked."),
            file_set("run-summaries", "One JSON file per training job: a dict from body name to the run summary (settings, three evaluator summaries, wall time and, for later panels, provenance).", "runs.zip", "application/json", "results/**/results.json"),
            file_set("analysis-outputs", "Small JSON outputs of analyses: morphology features, reviewed-version family statistics, early exploratory analyses (results/pivot/) and the retraining test analysis.", "runs.zip", "application/json", "results/*.json"),
            file_set("sim-records", "CPU MuJoCo records of 2026-10-05: zero-action rollouts, settle heights, random-action searches and training-like torso probes.", "runs.zip", "application/json", "sim-records/**/*.json"),
            file_set("gait-traces", "Joint-angle (qpos) and action traces of nine illustrative controllers (three bodies, three rules, one seed, later code revision) behind the gait figure.", "runs.zip", "application/x-npz", "gait-traces/*.traces.npz"),
            member_object("runs-table", "tables/runs.csv: one row per run of every panel.", "runs.zip", "tables/runs.csv"),
            member_object("features-table", "tables/morph_features.csv: structural features of the 100 UNIMAL and 60 non-evolved bodies.", "runs.zip", "tables/morph_features.csv"),
            file_set("training-logs", "Plain-text training logs.", "logs.zip", "text/plain", "**/*.log"),
            file_object("panels.zip", "Review-period panels (review-panels/): per-cell CSV tables, Humanoid evaluation curves, and the frozen reports, panel specifications, contracts and inventories of the Humanoid, Ant and Evolution Gym panels."),
            *(member_object(f"{rs}-cells-table", f"review-panels/{folder}/cells.csv: {desc}", "panels.zip",
                            f"review-panels/{folder}/cells.csv") for folder, rs, desc, _ in PANEL_TABLES),
            *(member_object(f"{rs}-progress-table", f"review-panels/{folder}/progress.csv: mean evaluation return of every cell at its six evaluation points.",
                            "panels.zip", f"review-panels/{folder}/progress.csv") for folder, rs, _, prog in PANEL_TABLES if prog),
            file_set("panel-records", "Frozen reports (report.json), analysis manifests, the composite analysis specification, panel specifications (root tokens and storage locations withheld) and contracts of the review-period panels.", "panels.zip", "application/json", "review-panels/**/*.json"),
            file_set("panel-inventories", "Frozen inventories: one JSON line per cell with its protocol settings.", "panels.zip", "application/jsonlines", "review-panels/**/*.jsonl"),
            file_set("panel-report-text", "Markdown views of the frozen reports, and the frozen summary of the Ant breadth panel.", "panels.zip", "text/markdown", "review-panels/**/*.md"),
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "runs",
                "name": "runs",
                "description": "One record per trained controller (3,306 runs over 11 panels).",
                "field": runs_fields(),
            },
            {
                "@type": "cr:RecordSet",
                "@id": "morphology_features",
                "name": "morphology_features",
                "description": "Structural features of each body, computed from its XML at the reset pose.",
                "field": [field("morphology_features", n, t, d, "features-table") for n, t, d in FEATURES],
            },
            *({
                "@type": "cr:RecordSet",
                "@id": f"{rs}_cells",
                "name": f"{rs}_cells",
                "description": f"One record per cell (morphology, protocol, seed, hardware class). {desc}",
                "field": [field(f"{rs}_cells", n, t, d, f"{rs}-cells-table") for n, t, d in PANEL_CELL_FIELDS],
            } for _, rs, desc, _ in PANEL_TABLES),
            *({
                "@type": "cr:RecordSet",
                "@id": f"{rs}_progress",
                "name": f"{rs}_progress",
                "description": f"One record per cell and evaluation point of {rs}_cells.",
                "field": [field(f"{rs}_progress", n, t, d, f"{rs}-progress-table") for n, t, d in PANEL_PROGRESS_FIELDS],
            } for _, rs, _, prog in PANEL_TABLES if prog),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archives", type=Path, help="directory holding runs.zip, logs.zip and panels.zip")
    ap.add_argument("--out", type=Path, default=HERE / "croissant.json")
    args = ap.parse_args()
    meta = build(args.archives)
    missing = [k for k in REQUIRED_CORE + REQUIRED_RAI if meta.get(k) in (None, "", [])]
    if missing:
        raise SystemExit(f"ERROR: missing required fields: {', '.join(missing)}")
    args.out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
