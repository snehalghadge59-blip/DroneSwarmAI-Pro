"""
main.py — DroneSwarmAI-Pro simulation entry point.

Usage
-----
# Run headless (terminal output only):
    python main.py

# 2D matplotlib renderer:
    python main.py --renderer 2d

# 3D matplotlib renderer:
    python main.py --renderer 3d

# Use RL mode:
    python main.py --mode rl

# Custom swarm size:
    python main.py --drones 30

# Limit frames:
    python main.py --steps 500
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from simulation.environment import Environment
from simulation.swarm       import Swarm
from utils.logger           import get_logger

log = get_logger("Main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DroneSwarmAI-Pro — Autonomous Drone Swarm Simulation"
    )
    p.add_argument("--renderer", choices=["none", "2d", "3d"],  default="2d",
                   help="Visualisation mode (default: 2d)")
    p.add_argument("--mode",     choices=["boids", "rl"],        default="boids",
                   help="Swarm AI mode (default: boids)")
    p.add_argument("--drones",   type=int, default=20,
                   help="Number of drones (default: 20)")
    p.add_argument("--steps",    type=int, default=0,
                   help="Max simulation steps; 0=infinite (default: 0)")
    p.add_argument("--no-leader",action="store_true",
                   help="Disable leader-follower behaviour")
    p.add_argument("--no-obstacles", action="store_true",
                   help="Spawn with zero obstacles")
    return p.parse_args()


def run_headless(swarm: Swarm, max_steps: int = 1000) -> None:
    """Run simulation without visualisation, printing progress."""
    log.info("Running headless for %s steps …",
             max_steps if max_steps else "∞")
    step = 0
    try:
        while max_steps == 0 or step < max_steps:
            metrics = swarm.step()
            if step % 50 == 0:
                log.info(
                    "Step %5d | alive=%d | coll=%d | targets=%d | avg_E=%.1f",
                    metrics["step"], metrics["n_alive"],
                    metrics["collisions"], metrics["targets"],
                    metrics["avg_energy"],
                )
            step += 1
    except KeyboardInterrupt:
        log.info("Interrupted at step %d", step)


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("DroneSwarmAI-Pro Starting")
    log.info("  drones=%d  mode=%s  renderer=%s",
             args.drones, args.mode, args.renderer)
    log.info("=" * 60)

    # Build environment
    n_static  = 0 if args.no_obstacles else 6
    n_dynamic = 0 if args.no_obstacles else 4
    env       = Environment(n_static=n_static, n_dynamic=n_dynamic)
    env.random_target()

    # Build swarm
    swarm = Swarm(
        env,
        n_drones     = args.drones,
        mode         = args.mode,
        leader_follow= not args.no_leader,
    )

    # Inject RL agent if needed
    if args.mode == "rl":
        from ai.rl_agent import DQNAgent
        agent = DQNAgent()
        agent.load("ai/dqn_model.pt")   # silently skips if not present
        swarm.rl_agent = agent
        log.info("RL agent attached (epsilon=%.2f)", agent.epsilon)

    # Launch renderer
    if args.renderer == "2d":
        try:
            from visualization.renderer_2d import Renderer2D
            r = Renderer2D(swarm, max_steps=args.steps, interval=50)
            r.run()
        except Exception as e:
            log.error("2D renderer failed: %s — falling back to headless", e)
            run_headless(swarm, args.steps or 500)

    elif args.renderer == "3d":
        try:
            from visualization.renderer_3d import Renderer3D
            r = Renderer3D(swarm, max_steps=args.steps, interval=60)
            r.run()
        except Exception as e:
            log.error("3D renderer failed: %s — falling back to headless", e)
            run_headless(swarm, args.steps or 500)

    else:
        run_headless(swarm, args.steps or 1000)

    # Final report
    stats = env.stats
    log.info("Simulation finished.")
    log.info("  Steps:       %d", stats["steps"])
    log.info("  Collisions:  %d", stats["total_collisions"])
    log.info("  Targets hit: %d", stats["targets_reached"])


if __name__ == "__main__":
    main()
