"""
swarm.py — Swarm manager: spawns drones, runs each tick, tracks metrics.

Supports two operating modes:
  - "boids"  : pure flocking (separation, alignment, cohesion)
  - "rl"     : boids blended with a trained RL agent's steering force

Leader-follower behaviour is implemented as a special case: drone 0 is
designated the leader; all others add a cohesion force toward the leader
in addition to standard Boids forces.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict

from simulation.drone       import Drone, MAX_SPEED
from simulation.environment import Environment
from utils.logger           import get_logger

log = get_logger("Swarm")


class Swarm:
    """
    Collection of Drone agents managed as a single unit.

    Parameters
    ----------
    env          : shared Environment (bounds, target, obstacles)
    n_drones     : initial swarm size
    mode         : "boids" or "rl"
    leader_follow: enable leader-follower sub-behaviour
    """

    def __init__(
        self,
        env:           Environment,
        n_drones:      int  = 20,
        mode:          str  = "boids",
        leader_follow: bool = True,
    ) -> None:
        self.env           = env
        self.mode          = mode
        self.leader_follow = leader_follow
        self.drones: List[Drone] = []
        self._spawn_drones(n_drones)

        # RL agent reference (injected later by train_rl / main)
        self.rl_agent = None

        # Per-step metric snapshots
        self.history: List[Dict] = []

        log.info("Swarm initialised: %d drones, mode=%s", n_drones, mode)

    # ── Initialisation ────────────────────────────────────────────────────

    def _spawn_drones(self, n: int) -> None:
        """Spawn n drones clustered near the world's centre."""
        centre = self.env.bounds / 2.0
        for i in range(n):
            offset = np.random.uniform(-8, 8, size=3)
            offset[2] = np.random.uniform(2, 10)          # stay above ground
            pos = np.clip(centre + offset, np.zeros(3), self.env.bounds)
            self.drones.append(Drone(position=pos, bounds=self.env.bounds))
        log.debug("Spawned %d drones near centre=%s", n, centre.tolist())

    # ── Drone management ─────────────────────────────────────────────────

    def add_drone(self) -> Drone:
        """Add one drone to a live swarm."""
        centre = self.env.bounds / 2.0
        pos = np.clip(centre + np.random.uniform(-10, 10, 3),
                      np.zeros(3), self.env.bounds)
        d = Drone(position=pos, bounds=self.env.bounds)
        self.drones.append(d)
        log.info("Added drone id=%d  (total=%d)", d.drone_id, len(self.drones))
        return d

    def remove_drone(self) -> None:
        """Remove the most recently added drone."""
        if self.drones:
            d = self.drones.pop()
            log.info("Removed drone id=%d  (total=%d)", d.drone_id, len(self.drones))

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self) -> Dict:
        """
        Advance all drones one simulation tick.

        Returns a metrics dict for this step.
        """
        self.env.step()           # advance obstacles etc.

        alive     = [d for d in self.drones if d.alive]
        obstacles = self.env.obstacles
        target    = self.env.target
        use_rl    = (self.mode == "rl")

        # ── Optional: RL agent picks action for each drone ────────────────
        if use_rl and self.rl_agent is not None:
            for drone in alive:
                state  = self._get_rl_state(drone)
                action = self.rl_agent.select_action(state)
                # Convert discrete action to force vector
                drone.rl_force = self._action_to_force(action)

        # ── Leader-follower: non-leaders bias toward leader ───────────────
        leader = alive[0] if alive else None

        for drone in alive:
            weights = {"sep": 1.5, "ali": 1.0, "coh": 1.0,
                       "obs": 3.0, "tgt": 2.0, "bnd": 2.0}

            if self.leader_follow and leader and drone.drone_id != leader.drone_id:
                # Additional cohesion pull toward leader
                ldir = leader.position - drone.position
                dist = np.linalg.norm(ldir)
                if dist > 0.1:
                    drone.apply_force(
                        (ldir / dist) * 0.3
                    )

            drone.update_position(
                all_drones = alive,
                obstacles  = obstacles,
                target     = target,
                weights    = weights,
                use_rl     = use_rl,
            )

            # Collision check
            if drone.check_collision(obstacles):
                self.env.stats["total_collisions"] += 1
                log.debug("Collision! drone id=%d step=%d",
                          drone.drone_id, self.env.step_count)

            # Target reached
            if drone.reached_target:
                self.env.stats["targets_reached"] += 1
                drone.reached_target = False   # reset for next target

        # Aggregate energy
        total_energy = sum(d.energy for d in alive)
        self.env.stats["total_energy_used"] = round(
            self.env.stats.get("total_energy_used", 0.0) +
            (len(alive) * 0.01), 2
        )

        metrics = {
            "step":         self.env.step_count,
            "n_alive":      len(alive),
            "collisions":   self.env.stats["total_collisions"],
            "targets":      self.env.stats["targets_reached"],
            "avg_energy":   round(total_energy / max(len(alive), 1), 2),
        }
        return metrics

    # ── RL helpers ────────────────────────────────────────────────────────

    def _get_rl_state(self, drone: Drone) -> np.ndarray:
        """
        Build a flat state vector for RL:
          [pos(3), vel(3), dist_to_target(1), dist_to_nearest_obs(1)]
        """
        pos     = drone.position / self.env.bounds           # normalised
        vel     = drone.velocity / MAX_SPEED
        tgt_d   = np.linalg.norm(self.env.target - drone.position) / \
                  np.linalg.norm(self.env.bounds)

        obs_d   = 1.0
        for obs in self.env.obstacles:
            d = (np.linalg.norm(drone.position - obs.position) - obs.radius)
            obs_d = min(obs_d, d / np.linalg.norm(self.env.bounds))

        return np.concatenate([pos, vel, [tgt_d, max(obs_d, 0.0)]])

    @staticmethod
    def _action_to_force(action: int) -> np.ndarray:
        """
        Map discrete action index to a 3-D steering force.

        Actions:
          0=+X  1=-X  2=+Y  3=-Y  4=+Z  5=-Z  6=hover
        """
        mapping = {
            0: np.array([ 1,  0,  0], dtype=np.float64),
            1: np.array([-1,  0,  0], dtype=np.float64),
            2: np.array([ 0,  1,  0], dtype=np.float64),
            3: np.array([ 0, -1,  0], dtype=np.float64),
            4: np.array([ 0,  0,  1], dtype=np.float64),
            5: np.array([ 0,  0, -1], dtype=np.float64),
            6: np.zeros(3,            dtype=np.float64),
        }
        return mapping.get(action, np.zeros(3))

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "drones": [d.to_dict() for d in self.drones],
            "mode":   self.mode,
            "env":    self.env.to_dict(),
        }

    @property
    def n_drones(self) -> int:
        return len(self.drones)

    def __repr__(self) -> str:
        return (f"Swarm(n={len(self.drones)}, mode={self.mode}, "
                f"step={self.env.step_count})")
