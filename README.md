# 🛸 DroneSwarmAI-Pro

**Autonomous Drone Swarm Simulation using AI — 3D + RL + Web + ROS**

A production-grade, research-level multi-agent drone swarm system combining swarm intelligence (Boids), Deep Reinforcement Learning (DQN), real-time 3D visualization, FastAPI web backend, and optional ROS2 integration.

---

## ✨ Features

| Category | Capabilities |
|---|---|
| **Swarm AI** | Boids (Separation, Alignment, Cohesion), Leader-Follower, Target Tracking |
| **RL Agent** | Deep Q-Network (DQN) with experience replay, target network, ε-greedy policy |
| **Visualization** | 2D Matplotlib top-down, 3D animated scene, Web canvas renderer |
| **Web Interface** | FastAPI REST + WebSocket, HTML5 mission control dashboard |
| **Streamlit** | Interactive dashboard with live charts and drone status table |
| **ROS Integration** | ROS2 drone nodes + swarm coordinator (graceful mock fallback) |
| **Obstacles** | Static spheres + sinusoidal dynamic obstacles |
| **Metrics** | Energy tracking, collision detection, target reach counting |
| **Logging** | Structured file + console logging via Python logging |

---

## 🏗️ Architecture

```
DroneSwarmAI-Pro/
│
├── app.py                    # Streamlit dashboard (streamlit run app.py)
├── main.py                   # CLI entry point     (python main.py)
│
├── simulation/
│   ├── environment.py        # World: bounds, target, obstacles, stats
│   ├── drone.py              # Agent: kinematics, Boids, RL blending
│   ├── swarm.py              # Manager: spawn, step, leader-follower, RL dispatch
│   └── obstacles.py          # Static & dynamic spherical obstacles
│
├── ai/
│   ├── boids.py              # Vectorised Boids (NumPy batch)
│   ├── rl_agent.py           # DQN agent (QNetwork, ReplayBuffer)
│   └── train_rl.py           # Training loop with reward shaping
│
├── visualization/
│   ├── renderer_2d.py        # Matplotlib top-down animated view
│   └── renderer_3d.py        # Matplotlib 3D animated scene
│
├── web/
│   ├── backend.py            # FastAPI server (REST + WebSocket)
│   └── frontend/index.html   # HTML5 mission control UI
│
├── ros/
│   ├── drone_node.py         # ROS2 node per drone (mock fallback)
│   └── swarm_controller.py   # ROS2 centralised swarm coordinator
│
├── utils/
│   ├── math_utils.py         # Vector ops: normalize, clamp, steer
│   └── logger.py             # Logging setup
│
├── logs/simulation.log       # Auto-generated runtime log
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-org/DroneSwarmAI-Pro.git
cd DroneSwarmAI-Pro
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For RL training with GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Running the Simulation

### Option A — Streamlit Dashboard (recommended for beginners)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with interactive controls, live 2D view, and metric charts.

---

### Option B — CLI with 2D Matplotlib

```bash
python main.py --renderer 2d --drones 25 --mode boids
```

**All CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--renderer` | `2d` | `none` \| `2d` \| `3d` |
| `--mode` | `boids` | `boids` \| `rl` |
| `--drones` | `20` | Number of agents (5–50) |
| `--steps` | `0` | Max steps (0 = infinite) |
| `--no-leader` | off | Disable leader-follower |
| `--no-obstacles` | off | Spawn without obstacles |

---

### Option C — Web UI + FastAPI

Start the backend:
```bash
uvicorn web.backend:app --host 0.0.0.0 --port 8000 --reload
```

Open `web/frontend/index.html` in a browser (or navigate to `http://localhost:8000/ui`).

Features of the web UI:
- ▶ Start / ■ Stop / ↺ Reset simulation
- Drone count slider (5–50)
- Mode toggle: BOIDS / RL
- Click-to-set target (crosshair mode)
- Velocity vectors, trails, obstacle toggle
- Live WebSocket canvas renderer at ~20 FPS
- Energy rings per drone
- Real-time metrics + event log

---

### Option D — 3D View

```bash
python main.py --renderer 3d --drones 30
```

Renders an auto-spinning 3D Matplotlib scene with wire-frame obstacles, drone scatter, and velocity info overlay.

---

## 🤖 Training the RL Agent

```bash
python -m ai.train_rl --episodes 500 --drones 5 --steps 400
```

The agent learns to navigate a single drone to a randomly repositioned target while avoiding obstacles. After training, the model is saved to `ai/dqn_model.pt` and automatically loaded when `--mode rl` is used.

**Reward shaping:**
- `+10` target reached
- `−5` collision
- `+Δd` distance improvement per step
- `−0.01` energy penalty per step

---

## 🤖 ROS2 Integration

Requires ROS2 Humble (or later) installed:

```bash
# Terminal 1 — swarm controller
ros2 run drone_swarm swarm_controller --n 10

# Terminal 2 — individual drone node
ros2 run drone_swarm drone_node --drone-id 0
```

Without ROS, both files run in mock mode automatically — no error, full terminal output.

---

## 🧠 How the Swarm Works

```
Each tick per drone:
  1. detect_neighbors()       — find drones within 6 world-units
  2. _separation()            — steer away from close neighbours
  3. _alignment()             — match average neighbour heading
  4. _cohesion()              — steer toward neighbour centre-of-mass
  5. avoid_obstacles()        — push away from obstacles (weight ×3)
  6. follow_target()          — steer toward global target
  7. _boundary_steer()        — soft wall repulsion at world edges
  8. [RL blend]               — 50/50 mix with DQN steering force
  9. velocity += Σforces      — Euler integration
 10. position += velocity     — move drone
 11. energy -= cost           — track efficiency
```

---

## 📊 Live Metrics

| Metric | Description |
|---|---|
| Drones Alive | Count of non-crashed drones |
| Step | Total simulation ticks elapsed |
| Collisions | Cumulative obstacle collisions |
| Targets Reached | Cumulative target arrivals |
| Avg Energy | Mean remaining energy across the swarm |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/state` | Full swarm JSON snapshot |
| GET | `/metrics` | Compact metrics dict |
| POST | `/start` | Resume simulation |
| POST | `/stop` | Pause simulation |
| POST | `/reset` | Full restart with config |
| POST | `/control` | Live-update drones/mode |
| POST | `/target` | Move target to {x,y,z} |
| WS | `/ws/state` | Live state at 20 FPS |
| GET | `/health` | API health check |

---

## 📸 Screenshots

> Run the simulation and capture screenshots here.

- `screenshots/web_dashboard.png`
- `screenshots/2d_view.png`
- `screenshots/3d_view.png`
- `screenshots/streamlit.png`

---

## 🔭 Future Scope

- [ ] **Formation flying** — V-formation, grid, diamond presets
- [ ] **Multi-target** — concurrent target assignments per sub-swarm
- [ ] **A* pathfinding** — global planner feeding local Boids
- [ ] **Gazebo integration** — full physics simulation
- [ ] **Real PX4 drones** — MAVLink bridge via MAVSDK-Python
- [ ] **GPU vectorisation** — CUDA-accelerated NumPy / CuPy batch Boids
- [ ] **Communication delay** — simulate packet loss and latency
- [ ] **Curriculum RL** — progressive difficulty during training
- [ ] **Unity3D frontend** — photorealistic 3D visualisation
- [ ] **Docker deployment** — single-container launch

---

## 📄 License

MIT License — see LICENSE for details.

---

## 🙏 Acknowledgements

- Craig Reynolds — original Boids model (1987)
- Mnih et al. — Deep Q-Networks (DQN) — DeepMind 2015
- FastAPI, Streamlit, Matplotlib, NumPy open-source communities
