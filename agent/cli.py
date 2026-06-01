"""
CLI entry: python -m agent <command>

Commands:
  run          — autonomous loop (AgentLoop + stub or vlm policy)
  probe        — one VLM step (GPU smoke test)
  state / step / interactive — debugging without full run
"""

from __future__ import annotations

import argparse
import sys

from agent.data_loader import repo_root
from agent.environment import World
from agent.clear_view import geometric_pole_in_clear_view
from agent.loop import AgentLoop
from agent.observations import print_observation, print_vlm_step_responses
from agent.policy import Policy, StubPolicy
from agent.types import POLE_TYPES, Action, ActionType, PoleType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pole-hunting agent CLI: stub planner or Qwen3-VL on GPU.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data/metadata (default: <repo>/data/metadata)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Autonomous agent run")
    _add_policy_args(run_p)
    run_p.add_argument("--max-steps", type=int, default=500)
    run_p.add_argument("--start-pano", type=str, default=None)
    run_p.add_argument("--json", action="store_true", help="Print observations as JSON")
    probe_p = sub.add_parser("probe", help="One VLM inference (smoke test on GPU VM)")
    _add_policy_args(probe_p, default_policy="vlm")
    probe_p.add_argument("--start-pano", type=str, default=None)

    state_p = sub.add_parser("state", help="Print initial or given state once")
    state_p.add_argument("--start-pano", type=str, default=None)
    state_p.add_argument("--json", action="store_true")

    step_p = sub.add_parser("step", help="Apply one action (manual)")
    step_p.add_argument("--start-pano", type=str, default=None)
    step_p.add_argument(
        "action",
        choices=["turn_left", "turn_right", "move", "classify_or_stop"],
    )
    step_p.add_argument("--pole-type", choices=POLE_TYPES, default=None)
    step_p.add_argument("--stop", action="store_true", help="Stop run after classify")

    interactive_p = sub.add_parser("interactive", help="Manual REPL: l/r/m/c/s/q")
    interactive_p.add_argument("--start-pano", type=str, default=None)

    return parser.parse_args(argv)


def _add_policy_args(parser: argparse.ArgumentParser, default_policy: str = "stub") -> None:
    parser.add_argument(
        "--policy",
        choices=["stub", "vlm"],
        default=default_policy,
        help="stub = graph planner; vlm = Qwen3-VL-4B-Instruct on this machine",
    )
    parser.add_argument(
        "--placeholder-type",
        choices=POLE_TYPES,
        default="lamp_post",
        help="Fallback pole type if the model omits pole_type on classify",
    )


def metadata_dir_from_arg(path: str | None):
    if path is None:
        return None
    from pathlib import Path

    p = Path(path)
    return p if p.name == "metadata" else p / "metadata"


def build_world(args: argparse.Namespace) -> World:
    return World.load(metadata_dir_from_arg(args.data_dir))


def build_policy(args: argparse.Namespace) -> Policy:
    if args.policy == "vlm":
        from agent.vlm_policy import VlmPolicy

        return VlmPolicy(fallback_type=args.placeholder_type)
    return StubPolicy(placeholder_type=args.placeholder_type)


def action_from_cli(name: str, pole_type: PoleType | None, stop: bool) -> Action:
    if name == "classify_or_stop":
        return Action(
            type=ActionType.CLASSIFY_OR_STOP,
            pole_type=pole_type,
            stop_after=stop,
        )
    return Action(type=ActionType(name))


def run_interactive(world: World, start_pano: str | None) -> int:
    state = world.initial_state(start_pano)
    loop = AgentLoop(world)

    print("Interactive agent — commands: l(turn left), r(turn right), m(move), c [type](classify), stop, s(state), q(quit)")
    print(f"Pole types: {', '.join(POLE_TYPES)}\n")

    while True:
        pole_in_clear_view = geometric_pole_in_clear_view(world, state)
        print_observation(world, state, pole_in_clear_view=pole_in_clear_view)
        if world.is_task_complete(state):
            print("\nAll poles classified.")
            return 0

        try:
            raw = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            continue
        if raw in {"q", "quit", "exit"}:
            return 0
        if raw in {"s", "state"}:
            continue

        if raw == "l":
            action = Action(type=ActionType.TURN_LEFT)
        elif raw == "r":
            action = Action(type=ActionType.TURN_RIGHT)
        elif raw == "m":
            action = Action(type=ActionType.MOVE)
        elif raw == "stop":
            action = Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)
        elif raw.startswith("c"):
            parts = raw.split()
            pole_type = parts[1] if len(parts) > 1 else None
            if pole_type and pole_type not in POLE_TYPES:
                print(f"Unknown type. Use one of: {', '.join(POLE_TYPES)}")
                continue
            action = Action(
                type=ActionType.CLASSIFY_OR_STOP,
                pole_type=pole_type,  # type: ignore[arg-type]
                stop_after=False,
            )
        else:
            print("Unknown command.")
            continue

        if state.pole_in_consideration is None and raw.startswith("c"):
            visible = world.poles_in_view(state)
            if visible:
                state.pole_in_consideration = visible[0].track_id
                print(f"Considering {visible[0].pole_id}")

        state, record = loop.step(state, action)
        print(record.message)

        if action.type == ActionType.CLASSIFY_OR_STOP and action.stop_after:
            return 0


def run_probe(world: World, policy: Policy, start_pano: str | None) -> int:
    from agent.vlm_policy import VlmPolicy

    if not isinstance(policy, VlmPolicy):
        print("probe requires --policy vlm", file=sys.stderr)
        return 1

    state = world.initial_state(start_pano)
    from agent.policy import apply_consideration

    from agent.vlm_policy import VlmPolicy

    if isinstance(policy, VlmPolicy):
        policy.begin_agent_step()
    state = apply_consideration(state, world, policy)
    pole_in_clear_view = policy.observe(world, state)
    print_observation(
        world, state, pole_in_clear_view=pole_in_clear_view, policy=policy
    )
    print("\nLoading VLM on this machine and running one step...", flush=True)
    action = policy.choose(world, state, pole_in_clear_view)
    print_vlm_step_responses(policy)
    print(f"\nmap image: {getattr(policy, 'last_map_image', None)}")
    print(f"street image: {getattr(policy, 'last_street_image', None)}")
    print(f"phase: {getattr(policy, 'last_phase', None)}")
    print(f"action: {action.type.value} pole_type={action.pole_type} stop_after={action.stop_after}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        world = build_world(args)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        print(f"Expected metadata under {repo_root() / 'data' / 'metadata'}", file=sys.stderr)
        return 1

    if args.command == "state":
        state = world.initial_state(getattr(args, "start_pano", None))
        print_observation(
            world,
            state,
            pole_in_clear_view=geometric_pole_in_clear_view(world, state),
            as_json=args.json,
        )
        return 0

    if args.command == "step":
        state = world.initial_state(args.start_pano)
        action = action_from_cli(args.action, args.pole_type, args.stop)
        loop = AgentLoop(world)
        state, record = loop.step(state, action)
        print(record.message)
        print_observation(
            world,
            state,
            pole_in_clear_view=geometric_pole_in_clear_view(world, state),
        )
        return 0

    if args.command == "probe":
        policy = build_policy(args)
        return run_probe(world, policy, args.start_pano)

    if args.command == "run":
        state = world.initial_state(args.start_pano)
        policy = build_policy(args)
        loop = AgentLoop(world, policy=policy)
        if args.policy == "vlm":
            print("VLM policy: agent and model run on this machine (see docs/GCP_VLM.md).", flush=True)
        final, history = loop.run(state, max_steps=args.max_steps, json_obs=args.json)
        print(f"\nFinished after {len(history)} steps. Classified {len(final.classified)}/{len(world.poles)}.")
        for track_id, pole_type in final.classified.items():
            pole = world.poles_by_track[track_id]
            print(f"  {pole.pole_id}: {pole_type}")
        return 0 if world.is_task_complete(final) else 1

    if args.command == "interactive":
        return run_interactive(world, args.start_pano)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
