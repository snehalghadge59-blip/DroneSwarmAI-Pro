"""
environment.py — World definition for the drone swarm simulation.

Holds world bounds, the target point, all obstacles, and global
simulation statistics.  Acts as the shared context passed through
the simulation loop.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from simulation.obstacles import Obstacle, make_default_obstacles


class Environment:
    """
    Simulation world container.

    Attributes
    ----------
    bounds       : np.ndarray [Xmax, Ymax, Zmax]
    target       : current 3-D target position
    obstacles    : list of Obstacle objects
    step_count   : total simulation steps elapsed
    stats        : running metrics (collisions, targets reached, etc.)
    """

    def __init__(
        self,
        bounds:    Optional[np.ndarray] = None,
        n_static:  int = 6,
        n_dynamic: int = 4,
    ) -> None:
        self.bounds = (bounds if bounds is not None
                       else np.array([60.0, 60.0, 30.0]))

        # Target starts at world centre, can be updated live
        self.target = self.bounds / 2.0

        self.obstacles: List[Obstacle] = make_default_obstacles(
            self.bounds, n_static=n_static, n_dynamic=n_dynamic
        )

        self.step_count   = 0
        self.stats: dict  = {
            "total_collisions":  0,
            "targets_reached":   0,
            "total_energy_used": 0.0,
            "steps":             0,
        }

    # ── Per-step update ───────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the environment one tick (moves dynamic obstacles)."""
        for obs in self.obstacles:
            obs.update()
        self.step_count += 1
        self.stats["steps"] = self.step_count

    # ── Helpers ───────────────────────────────────────────────────────────

    def set_target(self, pos: np.ndarray) -> None:
        """Move the target (e.g. user click or RL curriculum)."""
        self.target = np.clip(pos.astype(np.float64), np.zeros(3), self.bounds)

    def random_target(self) -> None:
        """Place target at a random location inside bounds."""
        margin = 4.0
        self.target = np.array([
            np.random.uniform(margin, self.bounds[0] - margin),
            np.random.uniform(margin, self.bounds[1] - margin),
            np.random.uniform(2.0,    self.bounds[2] - 2.0),
        ])

    def add_obstacle(self, obs: Obstacle) -> None:
        self.obstacles.append(obs)

    def remove_obstacle(self, index: int) -> None:
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "bounds":    self.bounds.tolist(),
            "target":    self.target.tolist(),
            "obstacles": [o.to_dict() for o in self.obstacles],
            "stats":     self.stats,
        }

    def __repr__(self) -> str:
        return (f"Environment(bounds={self.bounds.tolist()}, "
                f"obstacles={len(self.obstacles)}, "
                f"step={self.step_count})")
