"""
drone.py — Individual drone agent for the swarm simulation.

Each drone holds its own kinematic state and exposes methods for
physics integration, neighbour detection, obstacle avoidance and
target following.  All vectors are 3-D NumPy arrays [x, y, z].
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

from utils.math_utils import (
    normalize, clamp_magnitude, distance, steer_toward, random_unit_vector_3d
)

if TYPE_CHECKING:
    from simulation.obstacles import Obstacle


# ── Drone configuration defaults (can be overridden per-instance) ──────────
MAX_SPEED       = 4.0    # world-units / step
MAX_FORCE       = 0.4    # max steering force per step
PERCEPTION_R    = 6.0    # radius for neighbour detection
OBSTACLE_R      = 3.5    # radius for obstacle avoidance sensing
TARGET_R        = 1.2    # "close-enough" distance for target
DRAG            = 0.97   # velocity damping (0 < drag < 1)
ENERGY_COST     = 0.01   # energy units consumed per step of motion


class Drone:
    """
    Single autonomous drone agent.

    Attributes
    ----------
    drone_id    : unique integer identifier
    position    : np.ndarray shape (3,)
    velocity    : np.ndarray shape (3,)
    acceleration: np.ndarray shape (3,)
    energy      : remaining energy (starts at 100.0)
    alive       : False if collided fatally
    reached_target : True once target was reached
    """

    _id_counter: int = 0

    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        drone_id: Optional[int] = None,
        bounds: Optional[np.ndarray] = None,  # shape (3,) = [Xmax, Ymax, Zmax]
    ) -> None:
        # Unique ID
        if drone_id is None:
            Drone._id_counter += 1
            self.drone_id = Drone._id_counter
        else:
            self.drone_id = drone_id

        # World bounds (used for boundary steering)
        self.bounds = bounds if bounds is not None else np.array([50.0, 50.0, 30.0])

        # Kinematic state
        self.position     = position.copy().astype(np.float64) \
                            if position is not None \
                            else np.random.uniform(5, self.bounds - 5).astype(np.float64)
        self.velocity     = velocity.copy().astype(np.float64) \
                            if velocity is not None \
                            else random_unit_vector_3d() * 2.0
        self.acceleration = np.zeros(3, dtype=np.float64)

        # Book-keeping
        self.energy          = 100.0
        self.alive           = True
        self.reached_target  = False
        self.collision_count = 0

        # RL action (set externally by rl_agent; None = pure boids)
        self.rl_force: Optional[np.ndarray] = None

    # ── Internal force accumulator ─────────────────────────────────────────

    def apply_force(self, force: np.ndarray) -> None:
        """Accumulate a steering force into acceleration."""
        self.acceleration += force

    # ── Neighbour query ────────────────────────────────────────────────────

    def detect_neighbors(self, all_drones: List["Drone"]) -> List["Drone"]:
        """
        Return list of drones (excluding self) within PERCEPTION_R.
        Uses squared-distance for efficiency.
        """
        r2 = PERCEPTION_R ** 2
        neighbours = []
        for d in all_drones:
            if d.drone_id == self.drone_id or not d.alive:
                continue
            diff = d.position - self.position
            if np.dot(diff, diff) < r2:
                neighbours.append(d)
        return neighbours

    # ── Boids steering behaviours ──────────────────────────────────────────

    def _separation(self, neighbours: List["Drone"]) -> np.ndarray:
        """Steer away from close neighbours."""
        if not neighbours:
            return np.zeros(3)
        steer = np.zeros(3, dtype=np.float64)
        count = 0
        for n in neighbours:
            d = distance(self.position, n.position)
            if d < PERCEPTION_R * 0.5 and d > 1e-6:
                diff = (self.position - n.position) / (d * d)
                steer += diff
                count += 1
        if count > 0:
            steer /= count
        return clamp_magnitude(steer, MAX_FORCE)

    def _alignment(self, neighbours: List["Drone"]) -> np.ndarray:
        """Steer toward average heading of neighbours."""
        if not neighbours:
            return np.zeros(3)
        avg_vel = np.mean([n.velocity for n in neighbours], axis=0)
        return steer_toward(self.velocity, avg_vel, MAX_FORCE, MAX_SPEED)

    def _cohesion(self, neighbours: List["Drone"]) -> np.ndarray:
        """Steer toward centre-of-mass of neighbours."""
        if not neighbours:
            return np.zeros(3)
        centre = np.mean([n.position for n in neighbours], axis=0)
        desired = centre - self.position
        return steer_toward(self.velocity, desired, MAX_FORCE, MAX_SPEED)

    # ── Obstacle avoidance ─────────────────────────────────────────────────

    def avoid_obstacles(self, obstacles: List["Obstacle"]) -> np.ndarray:
        """
        Steer away from any obstacle within OBSTACLE_R.
        Returns combined avoidance force.
        """
        steer = np.zeros(3, dtype=np.float64)
        for obs in obstacles:
            d = distance(self.position, obs.position)
            clearance = d - obs.radius
            if clearance < OBSTACLE_R:
                away = self.position - obs.position
                weight = max(OBSTACLE_R - clearance, 0.1)
                steer += normalize(away) * weight
        return clamp_magnitude(steer, MAX_FORCE * 3)

    # ── Boundary steering ──────────────────────────────────────────────────

    def _boundary_steer(self) -> np.ndarray:
        """Push drone back inside world bounds with soft margin."""
        margin = 4.0
        force  = np.zeros(3, dtype=np.float64)
        for i in range(3):
            lo = margin
            hi = self.bounds[i] - margin
            if self.position[i] < lo:
                force[i] = (lo - self.position[i])
            elif self.position[i] > hi:
                force[i] = (hi - self.position[i])
        return clamp_magnitude(force * 0.5, MAX_FORCE)

    # ── Target following ───────────────────────────────────────────────────

    def follow_target(self, target: np.ndarray) -> np.ndarray:
        """Steer toward target position.  Sets reached_target flag."""
        desired = target - self.position
        dist = np.linalg.norm(desired)
        if dist < TARGET_R:
            self.reached_target = True
            return np.zeros(3)
        return steer_toward(self.velocity, desired, MAX_FORCE, MAX_SPEED)

    # ── Main update ────────────────────────────────────────────────────────

    def update_position(
        self,
        all_drones:  List["Drone"],
        obstacles:   List["Obstacle"],
        target:      Optional[np.ndarray] = None,
        weights:     Optional[dict]       = None,
        use_rl:      bool                 = False,
    ) -> None:
        """
        One physics step:
          1. Compute Boids + obstacle + boundary forces.
          2. Optionally blend in RL action force.
          3. Integrate velocity & position.
          4. Consume energy.
        """
        if not self.alive:
            return

        w = weights or {"sep": 1.5, "ali": 1.0, "coh": 1.0,
                        "obs": 3.0, "tgt": 2.0, "bnd": 2.0}

        neighbours = self.detect_neighbors(all_drones)

        sep   = self._separation(neighbours)   * w["sep"]
        ali   = self._alignment(neighbours)    * w["ali"]
        coh   = self._cohesion(neighbours)     * w["coh"]
        obs   = self.avoid_obstacles(obstacles)* w["obs"]
        bnd   = self._boundary_steer()         * w["bnd"]
        tgt   = (self.follow_target(target)    * w["tgt"]
                 if target is not None else np.zeros(3))

        total = sep + ali + coh + obs + bnd + tgt

        if use_rl and self.rl_force is not None:
            # Blend RL steering with Boids at 50/50
            total = 0.5 * total + 0.5 * clamp_magnitude(self.rl_force, MAX_FORCE)

        self.apply_force(total)

        # Integrate
        self.velocity     += self.acceleration
        self.velocity     *= DRAG
        self.velocity      = clamp_magnitude(self.velocity, MAX_SPEED)
        self.position     += self.velocity
        self.acceleration  = np.zeros(3, dtype=np.float64)

        # Clamp to world
        self.position = np.clip(self.position, np.zeros(3), self.bounds)

        # Energy
        speed = np.linalg.norm(self.velocity)
        self.energy = max(0.0, self.energy - ENERGY_COST * (0.5 + speed / MAX_SPEED))

    # ── Collision check ────────────────────────────────────────────────────

    def check_collision(self, obstacles: List["Obstacle"]) -> bool:
        """Return True if drone has physically entered any obstacle."""
        for obs in obstacles:
            if distance(self.position, obs.position) < obs.radius * 0.9:
                self.collision_count += 1
                return True
        return False

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Compact serialisable snapshot for logging / web API."""
        return {
            "id":       self.drone_id,
            "pos":      self.position.tolist(),
            "vel":      self.velocity.tolist(),
            "energy":   round(self.energy, 2),
            "alive":    self.alive,
            "reached":  self.reached_target,
            "col":      self.collision_count,
        }

    def __repr__(self) -> str:
        p = self.position
        return (f"Drone(id={self.drone_id}, "
                f"pos=[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}], "
                f"energy={self.energy:.1f})")
