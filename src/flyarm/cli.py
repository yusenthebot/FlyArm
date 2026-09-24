"""Explicit downloads and bounded experiments; nothing runs on import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyarm.assets import fetch_arm, fetch_data
from flyarm.config import ExperimentConfig, PickPlaceConfig
from flyarm.graph import prepare_graph

DEFAULT_PACK = "data/whole_brain/malecns-v1.0-c3"
DEFAULT_ANNOTATIONS = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"


def _whole_brain(args: argparse.Namespace) -> None:
    # MLX is imported lazily so the subgraph commands keep working on non-Apple hosts.
    if args.brain_command == "compile":
        from flyarm.whole_brain.compiler import compile_connectome

        pack = compile_connectome(args.raw, args.output, min_contacts=args.min_contacts)
        summary = {k: v for k, v in pack.manifest.items() if k != "sources"}
        print(json.dumps({**summary, "fingerprint": pack.fingerprint()}, indent=2))
    elif args.brain_command == "check":
        from flyarm.experiment import save_json
        from flyarm.whole_brain.diagnostics import go_no_go

        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = go_no_go(args.pack)
        save_json(args.output, report)
        print(json.dumps({"go": report["go"], **report["checks"]}, indent=2))
    elif args.brain_command == "record":
        from flyarm.whole_brain.experiment import load_trained_policy, record_rollouts, run_name

        task, policy = load_trained_policy(
            args.run, args.kind, args.seed, args.pack, args.model, replicate=args.replicate
        )
        try:
            seeds = task.seeds("test", args.episodes)
            checkpoint = run_name(args.kind, args.seed, args.replicate)
            path = args.output or args.run / checkpoint / "rollout.mp4"
            names = {
                "connectome": "Complete MaleCNS (166,700 neurons)",
                "shuffled": "Degree-matched shuffled CNS",
                "gru": "GRU, parameter-matched",
            }
            replicate = f" #{args.replicate + 1}" if args.kind == "shuffled" else ""
            title = f"{names[args.kind]}{replicate} · {task.name} · seed {args.seed}"
            record_rollouts(task, policy, seeds, path, title=title)
        finally:
            task.close()
        print(path)
    elif args.brain_command == "serve":
        from flyarm.whole_brain.live import serve_whole_brain

        serve_whole_brain(
            args.run,
            args.pack,
            args.annotations,
            args.model,
            args.ui,
            host=args.host,
            port=args.port,
            seed=args.seed,
            episode=args.episode,
        )
    elif args.brain_command == "run":
        from flyarm.config import WholeBrainConfig
        from flyarm.whole_brain.experiment import run_whole_brain_experiment

        config = WholeBrainConfig.model_validate_json(args.config.read_text())
        result = run_whole_brain_experiment(args.pack, args.model, args.output, config)
        print(json.dumps(result["evidence"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyArm connectome control research MVP")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("fetch", help="Download pinned MaleCNS data (~1.1 GB) and Panda assets")
    setup.add_argument("--data", type=Path, default=Path("data/raw"))
    setup.add_argument("--assets", type=Path, default=Path("assets/menagerie"))
    prepare = sub.add_parser("prepare", help="Build a deterministic measured subgraph")
    prepare.add_argument("--raw", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output", type=Path, default=Path("data/graphs/malecns-256-v1.npz"))
    prepare.add_argument("--nodes", type=int, default=256)
    run = sub.add_parser(
        "run", help="Collect, imitate, evaluate, and record actual MuJoCo rollouts"
    )
    run.add_argument("--config", type=Path, default=Path("configs/reach.json"))
    run.add_argument("--graph", type=Path, default=Path("data/graphs/malecns-256-v1.npz"))
    run.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    run.add_argument("--output", type=Path, required=True)
    pick = sub.add_parser(
        "pick-place", help="Run restricted-interface, real-contact pick-and-place experiments"
    )
    pick.add_argument("--config", type=Path, default=Path("configs/pick-place.json"))
    pick.add_argument("--graph", type=Path, default=Path("data/graphs/malecns-256-v1.npz"))
    pick.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    pick.add_argument("--output", type=Path, required=True)
    brain = sub.add_parser("whole-brain", help="B1a: full MaleCNS rate controller (MLX)")
    brain_sub = brain.add_subparsers(dest="brain_command", required=True)
    compile_pack = brain_sub.add_parser("compile", help="Compile the full connectome CSR pack")
    compile_pack.add_argument("--raw", type=Path, default=Path("data/raw"))
    compile_pack.add_argument("--output", type=Path, default=Path(DEFAULT_PACK))
    compile_pack.add_argument("--min-contacts", type=int, default=3)
    check = brain_sub.add_parser("check", help="Go/no-go: load, determinism, bypass, speed")
    check.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    check.add_argument("--output", type=Path, required=True)
    brain_run = brain_sub.add_parser("run", help="Train and evaluate B1a controllers")
    brain_run.add_argument("--config", type=Path, default=Path("configs/whole-brain-reach.json"))
    brain_run.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    brain_run.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    brain_run.add_argument("--output", type=Path, required=True)
    record = brain_sub.add_parser("record", help="Record MP4 rollouts of a saved checkpoint")
    record.add_argument("--run", type=Path, required=True)
    record.add_argument("--kind", choices=["connectome", "shuffled", "gru"], default="connectome")
    record.add_argument("--seed", type=int, default=0)
    record.add_argument("--replicate", type=int, default=0, help="shuffle replicate index")
    record.add_argument("--episodes", type=int, default=3)
    record.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    record.add_argument("--output", type=Path, help="MP4 path (default: inside the run)")
    record.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    brain_serve = brain_sub.add_parser("serve", help="Live UI driven by a B1a checkpoint")
    brain_serve.add_argument("--run", type=Path, required=True)
    brain_serve.add_argument("--seed", type=int, default=0, help="training seed of the checkpoint")
    brain_serve.add_argument("--episode", type=int, default=60000, help="episode seed shown")
    brain_serve.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    brain_serve.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"),
    )
    brain_serve.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    brain_serve.add_argument("--ui", type=Path, default=Path("ui/dist"))
    brain_serve.add_argument("--host", default="127.0.0.1")
    brain_serve.add_argument("--port", type=int, default=8769)
    leg = sub.add_parser("flyleg", help="B2: the arm as the fly's left front leg (FrankaKitchen)")
    leg_sub = leg.add_subparsers(dest="leg_command", required=True)
    leg_run = leg_sub.add_parser("run", help="Train and evaluate on a D4RL kitchen split")
    leg_run.add_argument("--config", type=Path, required=True)
    leg_run.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    leg_run.add_argument("--annotations", type=Path, default=Path(DEFAULT_ANNOTATIONS))
    leg_run.add_argument("--output", type=Path, required=True)
    leg_record = leg_sub.add_parser("record", help="Labelled kitchen rollouts of saved checkpoints")
    leg_record.add_argument("--run", type=Path, required=True)
    leg_record.add_argument("--seed", type=int, default=0)
    leg_record.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    leg_record.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    leg_record.add_argument("--annotations", type=Path, default=Path(DEFAULT_ANNOTATIONS))
    leg_record.add_argument("--output", type=Path, required=True)
    manipulation = sub.add_parser(
        "manipulation", help="Articulated multi-step manipulation on the frozen connectome"
    )
    manipulation_sub = manipulation.add_subparsers(dest="manipulation_command", required=True)
    imitate = manipulation_sub.add_parser(
        "imitate", help="Behavior cloning and DAgger from the scripted teacher, then evaluation"
    )
    imitate.add_argument("--config", type=Path, required=True)
    imitate.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    imitate.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    imitate.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    imitate.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report", help="Aggregate completed runs into a Markdown report")
    report.add_argument("--output", type=Path, required=True)
    board = sub.add_parser("dashboard", help="Local run history, curves, logs and rollouts")
    board.add_argument("--host", default="127.0.0.1")
    board.add_argument("--port", type=int, default=8780)
    rl = sub.add_parser("rl", help="Reinforcement learning with batched MuJoCo (mjbatch)")
    rl_sub = rl.add_subparsers(dest="rl_command", required=True)
    ppo = rl_sub.add_parser("ppo", help="PPO fine-tuning of a pick-place checkpoint's decoder")
    ppo.add_argument("--config", type=Path, required=True)
    ppo.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    ppo.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    ppo.add_argument("--output", type=Path, required=True)
    kitchen_ppo = rl_sub.add_parser(
        "kitchen", help="PPO on the batched FrankaKitchen benchmark (reward, not demonstrations)"
    )
    kitchen_ppo.add_argument("--config", type=Path, required=True)
    kitchen_ppo.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    kitchen_ppo.add_argument("--output", type=Path, required=True)
    manipulation_ppo = rl_sub.add_parser(
        "manipulation", help="PPO on the batched manipulation benchmark from an imitation run"
    )
    manipulation_ppo.add_argument("--config", type=Path, required=True)
    manipulation_ppo.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    manipulation_ppo.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    manipulation_ppo.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    manipulation_ppo.add_argument("--output", type=Path, required=True)
    rl_record = rl_sub.add_parser("record", help="Before/after videos of a PPO run")
    rl_record.add_argument("--run", type=Path, required=True)
    rl_record.add_argument("--variant", default="nominal")
    rl_record.add_argument("--seeds", type=int, nargs="+", default=[60000, 60001])
    rl_record.add_argument("--iteration", type=int, help="PPO checkpoint (default: last)")
    rl_record.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    rl_record.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    rl_watch = rl_sub.add_parser(
        "watch", help="Keep one short rollout clip of the newest checkpoint of a running PPO run"
    )
    rl_watch.add_argument("--run", type=Path, required=True)
    rl_watch.add_argument("--variant", default="nominal")
    rl_watch.add_argument("--episodes", type=int, default=2)
    rl_watch.add_argument("--poll-seconds", type=float, default=60.0)
    rl_watch.add_argument("--once", action="store_true", help="Record once and exit")
    rl_watch.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    rl_watch.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    rl_watch.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    leg_serve = leg_sub.add_parser("serve", help="Live kitchen UI driven by a B2 checkpoint")
    leg_serve.add_argument("--run", type=Path, required=True)
    leg_serve.add_argument("--seed", type=int, default=0)
    leg_serve.add_argument("--episode", type=int, default=0)
    leg_serve.add_argument("--pack", type=Path, default=Path(DEFAULT_PACK))
    leg_serve.add_argument("--annotations", type=Path, default=Path(DEFAULT_ANNOTATIONS))
    leg_serve.add_argument("--ui", type=Path, default=Path("ui/dist"))
    leg_serve.add_argument("--host", default="127.0.0.1")
    leg_serve.add_argument("--port", type=int, default=8770)
    serve = sub.add_parser("serve", help="Launch the real-time causal simulator and 3D UI")
    serve.add_argument("--run", type=Path, required=True)
    serve.add_argument("--graph", type=Path, default=Path("data/graphs/malecns-256-v1.npz"))
    serve.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"),
    )
    serve.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    serve.add_argument("--ui", type=Path, default=Path("ui/dist"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8768)
    serve.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_data(args.data)
        print(fetch_arm(args.assets))
    elif args.command == "prepare":
        graph = prepare_graph(args.raw, args.output, args.nodes)
        print(json.dumps({k: v for k, v in graph.metadata.items() if k != "sources"}, indent=2))
    elif args.command == "run":
        from flyarm.experiment import run_experiment

        reach_config = ExperimentConfig.model_validate_json(args.config.read_text())
        result = run_experiment(args.graph, args.model, args.output, reach_config)
        print(f"Complete: {args.output}; {len(result['models'])} trained models")
    elif args.command == "pick-place":
        from flyarm.pick_place_experiment import run_pick_place_experiment

        pick_config = PickPlaceConfig.model_validate_json(args.config.read_text())
        result = run_pick_place_experiment(args.graph, args.model, args.output, pick_config)
        print(f"Complete: {args.output}; {len(result['models'])} trained models")
    elif args.command == "whole-brain":
        _whole_brain(args)
    elif args.command == "dashboard":
        from flyarm.dashboard.server import serve_dashboard

        serve_dashboard(Path.cwd(), args.host, args.port)
    elif args.command == "rl" and args.rl_command == "kitchen":
        from flyarm.config import KitchenPPOConfig
        from flyarm.rl.ppo_kitchen import run_kitchen_ppo

        kitchen_config = KitchenPPOConfig.model_validate_json(args.config.read_text())
        kitchen_outcome = run_kitchen_ppo(kitchen_config, args.pack, args.output)
        print(json.dumps({k: kitchen_outcome[k] for k in ("base", "best")}, indent=2))
    elif args.command == "manipulation":
        from flyarm.config import ManipulationImitationConfig
        from flyarm.manipulation.imitation import run_manipulation_imitation

        imitation_config = ManipulationImitationConfig.model_validate_json(args.config.read_text())
        imitation = run_manipulation_imitation(
            args.pack, args.model, args.output, imitation_config, args.asset_root
        )
        print(json.dumps({m["kind"]: m["evaluation"] for m in imitation["models"]}, indent=2))
    elif args.command == "rl" and args.rl_command == "manipulation":
        from flyarm.config import ManipulationPPOConfig
        from flyarm.rl.ppo_manipulation import run_manipulation_ppo

        manipulation_config = ManipulationPPOConfig.model_validate_json(args.config.read_text())
        manipulation_outcome = run_manipulation_ppo(
            manipulation_config, args.pack, args.model, args.output, args.asset_root
        )
        print(json.dumps({k: manipulation_outcome[k] for k in ("base", "best")}, indent=2))
    elif args.command == "rl" and args.rl_command == "watch":
        from flyarm.progress import watch

        watch(
            args.run,
            args.pack,
            args.model,
            episodes=args.episodes,
            poll_seconds=args.poll_seconds,
            variant=args.variant,
            once=args.once,
            asset_root=args.asset_root,
        )
    elif args.command == "rl" and args.rl_command == "record":
        from flyarm.rl.record import record_before_after

        print(
            record_before_after(
                args.run,
                args.pack,
                args.model,
                variant=args.variant,
                seeds=args.seeds,
                iteration=args.iteration,
            )
        )
    elif args.command == "rl":
        from flyarm.config import PPOConfig
        from flyarm.rl.ppo import run_ppo

        ppo_config = PPOConfig.model_validate_json(args.config.read_text())
        outcome = run_ppo(ppo_config, args.pack, args.model, args.output)
        print(json.dumps({k: outcome[k] for k in ("base", "best")}, indent=2))
    elif args.command == "report":
        from flyarm.report import build_report

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(build_report(Path.cwd()))
        print(args.output)
    elif args.command == "flyleg" and args.leg_command == "serve":
        from flyarm.flyleg.live import serve_flyleg

        serve_flyleg(
            args.run,
            args.seed,
            args.pack,
            args.annotations,
            args.ui,
            host=args.host,
            port=args.port,
            episode=args.episode,
        )
    elif args.command == "flyleg" and args.leg_command == "record":
        from flyarm.flyleg.record import record_kitchen

        manifest = record_kitchen(
            args.run, args.seed, args.episodes, args.pack, args.annotations, args.output
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "flyleg":
        from flyarm.config import FlyLegConfig
        from flyarm.flyleg.experiment import run_flyleg_experiment

        leg_config = FlyLegConfig.model_validate_json(args.config.read_text())
        outcome = run_flyleg_experiment(args.pack, args.annotations, args.output, leg_config)
        print(json.dumps(outcome["summary"], indent=2))
    else:
        from flyarm.live import serve_live

        serve_live(
            args.graph,
            args.annotations,
            args.model,
            args.run,
            args.ui,
            host=args.host,
            port=args.port,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
