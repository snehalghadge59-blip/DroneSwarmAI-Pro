"""
app.py — Streamlit dashboard for DroneSwarmAI-Pro.

Provides interactive controls and live chart visualisation
without requiring a separate FastAPI server.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import streamlit as st

from simulation.environment import Environment
from simulation.swarm       import Swarm

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "DroneSwarmAI Pro",
    page_icon   = "🛸",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body, .stApp { background-color: #07090f; color: #cdd6f4; }
  .metric-label { color: #4a5568 !important; font-size: 0.7rem !important; }
  .metric-value { color: #00e5ff !important; }
  .stButton>button {
    background: transparent;
    border: 1px solid #1e2a3a;
    color: #cdd6f4;
    border-radius: 6px;
    letter-spacing: 1px;
    font-family: monospace;
  }
  .stButton>button:hover {
    border-color: #00e5ff;
    color: #00e5ff;
  }
  .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────

def _init_swarm(n_drones: int = 20, mode: str = "boids") -> None:
    env   = Environment()
    env.random_target()
    swarm = Swarm(env, n_drones=n_drones, mode=mode)
    if mode == "rl":
        from ai.rl_agent import DQNAgent
        agent = DQNAgent()
        agent.load("ai/dqn_model.pt")
        swarm.rl_agent = agent
    st.session_state["swarm"]   = swarm
    st.session_state["running"] = False
    st.session_state["history_n"]  = []
    st.session_state["history_col"]= []
    st.session_state["history_tgt"]= []

if "swarm" not in st.session_state:
    _init_swarm()

swarm: Swarm = st.session_state["swarm"]

# ── Sidebar controls ──────────────────────────────────────────────────────
st.sidebar.markdown("## 🛸 DroneSwarmAI Pro")
st.sidebar.markdown("---")

n_drones = st.sidebar.slider("Drones", 5, 50, 20, 1)
mode     = st.sidebar.selectbox("Mode", ["boids", "rl"])
speed    = st.sidebar.slider("Speed (steps/render)", 1, 10, 3)
st.sidebar.markdown("---")

col1, col2 = st.sidebar.columns(2)
if col1.button("▶ START"):
    st.session_state["running"] = True
if col2.button("■ STOP"):
    st.session_state["running"] = False

if st.sidebar.button("↺ RESET"):
    _init_swarm(n_drones, mode)
    swarm = st.session_state["swarm"]
    st.rerun()

st.sidebar.markdown("---")
show_vel     = st.sidebar.checkbox("Velocity Vectors", True)
show_obs     = st.sidebar.checkbox("Obstacles",        True)
leader_follow= st.sidebar.checkbox("Leader-Follower",  True)

st.sidebar.markdown("---")
tx = st.sidebar.number_input("Target X", 0.0, 60.0, 30.0, 1.0)
ty = st.sidebar.number_input("Target Y", 0.0, 60.0, 30.0, 1.0)
tz = st.sidebar.number_input("Target Z", 0.0, 30.0, 15.0, 1.0)
if st.sidebar.button("📍 Set Target"):
    swarm.env.set_target(np.array([tx, ty, tz]))

# ── Main content ──────────────────────────────────────────────────────────
st.title("🛸 Drone Swarm Mission Control")

# Metrics row
m1, m2, m3, m4, m5 = st.columns(5)
stats = swarm.env.stats
alive = [d for d in swarm.drones if d.alive]

m1.metric("Drones Alive",    len(alive))
m2.metric("Simulation Step", stats.get("steps", 0))
m3.metric("Collisions",      stats.get("total_collisions", 0))
m4.metric("Targets Reached", stats.get("targets_reached", 0))
avg_e = (sum(d.energy for d in alive) / max(len(alive), 1))
m5.metric("Avg Energy",      f"{avg_e:.1f}%")

st.markdown("---")

# Plot columns
plot_col, chart_col = st.columns([2, 1])

# ── 2D top-down canvas via matplotlib ────────────────────────────────────
with plot_col:
    st.subheader("Top-Down View")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    bounds = swarm.env.bounds
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#07090f")
    ax.set_facecolor("#0a0e1a")
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_aspect("equal")

    # Obstacles
    if show_obs:
        for obs in swarm.env.obstacles:
            c = plt.Circle((obs.position[0], obs.position[1]),
                           obs.radius, color="#ff6e40", alpha=0.4)
            ax.add_patch(c)

    # Target
    t = swarm.env.target
    ax.scatter([t[0]], [t[1]], s=200, c="#ff4081", marker="X", zorder=7)
    ax.annotate("TARGET", (t[0], t[1]), color="#ff4081",
                fontsize=7, ha="center", va="bottom")

    # Drones
    for i, drone in enumerate(alive):
        px, py = drone.position[0], drone.position[1]
        color  = "#ffd600" if i == 0 else "#00e5ff"
        size   = 80 if i == 0 else 40
        ax.scatter([px], [py], s=size, c=color, zorder=5, alpha=0.9)
        if show_vel:
            vx, vy = drone.velocity[0], drone.velocity[1]
            ax.annotate("", xy=(px+vx*2, py+vy*2), xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color="#69f0ae",
                                        lw=0.8, alpha=0.6))

    ax.tick_params(colors="#4a5568", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#1e2a3a")
    ax.set_xlabel("X", color="#4a5568")
    ax.set_ylabel("Y", color="#4a5568")
    ax.set_title(f"Mode: {swarm.mode.upper()}  |  Step: {stats.get('steps',0)}",
                 color="#e2e8f0", fontsize=10)

    legend_handles = [
        mpatches.Patch(color="#00e5ff", label="Drone"),
        mpatches.Patch(color="#ffd600", label="Leader"),
        mpatches.Patch(color="#ff4081", label="Target"),
        mpatches.Patch(color="#ff6e40", label="Obstacle"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              facecolor="#1a2035", labelcolor="#e2e8f0", fontsize=7,
              framealpha=0.8)

    fig_placeholder = st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# ── Live charts ───────────────────────────────────────────────────────────
with chart_col:
    st.subheader("Live Metrics")

    hist_n   = st.session_state["history_n"]
    hist_col = st.session_state["history_col"]
    hist_tgt = st.session_state["history_tgt"]

    if len(hist_n) > 0:
        import pandas as pd
        df = pd.DataFrame({
            "Drones Alive":    hist_n[-100:],
            "Collisions":      hist_col[-100:],
            "Targets Reached": hist_tgt[-100:],
        })
        st.line_chart(df, height=250)

    # Drone data table
    st.subheader("Drone Status")
    import pandas as pd
    rows = []
    for d in swarm.drones[:10]:
        rows.append({
            "ID":    d.drone_id,
            "X":     round(d.position[0], 1),
            "Y":     round(d.position[1], 1),
            "Z":     round(d.position[2], 1),
            "E %":   round(d.energy, 1),
            "Col":   d.collision_count,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

# ── Simulation tick ───────────────────────────────────────────────────────

if st.session_state["running"]:
    # Sync live settings
    swarm.mode          = mode
    swarm.leader_follow = leader_follow
    current_n = swarm.n_drones
    diff      = n_drones - current_n
    if diff > 0:
        for _ in range(diff):   swarm.add_drone()
    elif diff < 0:
        for _ in range(-diff):  swarm.remove_drone()

    for _ in range(speed):
        swarm.step()

    # History tracking
    st.session_state["history_n"  ].append(len(alive))
    st.session_state["history_col"].append(stats.get("total_collisions", 0))
    st.session_state["history_tgt"].append(stats.get("targets_reached",  0))

    time.sleep(0.05)
    st.rerun()
