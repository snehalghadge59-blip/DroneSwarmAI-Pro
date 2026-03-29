"""
obstacles.py — Static and dynamic obstacles for the simulation world.

Each obstacle is a sphere defined by a centre position and radius.
Dynamic obstacles follow a sinusoidal patrol path.
"""

from __future__ import annotations

import numpy as np
from typing import List


class Obstacle:
    """
    Spherical obstacle (static or dynamic).

    Parameters
    ----------
    position : initial centre (x, y, z)
    radius   : collision / avoidance radius
    dynamic  : if True, the obstacle patrols on a sine path
    speed    : patrol speed (world-units / step)
    axis     : patrol axis index 0=X, 1=Y, 2=Z
    amplitude: patrol half-extent
    """

    def __init__(
        self,
        position: np.ndarray,
        radius: float = 2.0,
        dynamic: bool = False,
        speed: float = 0.05,
        axis: int = 0,
        amplitude: float = 8.0,
    ) -> None:
        self.position   = position.astype(np.float64)
        self._origin    = position.astype(np.float64)
        self.radius     = float(radius)
        self.dynamic    = dynamic
        self.speed      = speed
        self.axis       = axis
        self.amplitude  = amplitude
        self._phase     = float(np.random.uniform(0, 2 * np.pi))
        self._tick      = 0

    def update(self) -> None:
        """Move dynamic obstacle one step along its patrol path."""
        if not self.dynamic:
            return
        self._tick += 1
        offset = self.amplitude * np.sin(self.speed * self._tick + self._phase)
        self.position = self._origin.copy()
        self.position[self.axis] += offset

    def to_dict(self) -> dict:
        return {
            "pos":     self.position.tolist(),
            "radius":  self.radius,
            "dynamic": self.dynamic,
        }

    def __repr__(self) -> str:
        p = self.position
        return (f"Obstacle(pos=[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}], "
                f"r={self.radius}, dynamic={self.dynamic})")


def make_default_obstacles(bounds: np.ndarray, n_static: int = 5,
                            n_dynamic: int = 3) -> List[Obstacle]:
    """
    Generate a mix of static and dynamic obstacles placed randomly within
    the world bounds, avoiding the corners (drone spawn zones).
    """
    obstacles: List[Obstacle] = []
    margin = 8.0

    for _ in range(n_static):
        pos = np.array([
            np.random.uniform(margin, bounds[0] - margin),
            np.random.uniform(margin, bounds[1] - margin),
            np.random.uniform(3.0, bounds[2] - 3.0),
        ])
        r = np.random.uniform(1.5, 3.5)
        obstacles.append(Obstacle(pos, radius=r, dynamic=False))

    for _ in range(n_dynamic):
        pos = np.array([
            np.random.uniform(margin, bounds[0] - margin),
            np.random.uniform(margin, bounds[1] - margin),
            np.random.uniform(3.0, bounds[2] - 3.0),
        ])
        r     = np.random.uniform(1.0, 2.5)
        axis  = np.random.randint(0, 2)     # patrol X or Y
        amp   = np.random.uniform(5.0, 12.0)
        spd   = np.random.uniform(0.03, 0.08)
        obstacles.append(Obstacle(pos, radius=r, dynamic=True,
                                  speed=spd, axis=axis, amplitude=amp))

    return obstacles
