"""Explicit downloads and bounded experiments; nothing runs on import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyarm.assets import fetch_arm, fetch_data
from flyarm.config import ExperimentConfig, PickPlaceConfig
from flyarm.graph import prepare_graph


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
