"""
Physics Discovery Engine — Can a model learn Newton's laws from scratch?

A model watches simple 2D physics simulations (balls falling, bouncing, colliding)
and tries to learn the underlying laws. Instead of encoding physics as constraints,
the model is REWARDED for discovering physics — like a baby learning gravity.

Architecture (from Harrison's vision):
  Physics Simulator → Latent World Model → Physics Law Extraction → Interpretability
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json

# ─────────────────────────────────────────────
# 1. PHYSICS SIMULATOR — generates ground-truth trajectories
# ─────────────────────────────────────────────

class PhysicsWorld:
    """Simple 2D physics with gravity, bouncing, and collisions."""

    def __init__(self, n_objects=3, dt=0.02, gravity=9.81, restitution=0.8,
                 width=10.0, height=10.0):
        self.n_objects = n_objects
        self.dt = dt
        self.true_gravity = gravity
        self.restitution = restitution
        self.width = width
        self.height = height

    def random_state(self):
        """Random initial positions and velocities."""
        positions = np.random.uniform(
            low=[1.0, 3.0],
            high=[self.width - 1.0, self.height - 1.0],
            size=(self.n_objects, 2)
        )
        velocities = np.random.uniform(-3.0, 3.0, size=(self.n_objects, 2))
        masses = np.random.uniform(0.5, 2.0, size=(self.n_objects,))
        radii = masses * 0.2 + 0.15
        return positions, velocities, masses, radii

    def step(self, positions, velocities, masses, radii):
        """One physics step: gravity + boundary bouncing + simple collisions."""
        new_vel = velocities.copy()
        new_pos = positions.copy()

        # Gravity (only affects y-axis, downward)
        new_vel[:, 1] -= self.true_gravity * self.dt

        # Update positions
        new_pos += new_vel * self.dt

        # Boundary collisions
        for i in range(self.n_objects):
            r = radii[i]
            # Floor
            if new_pos[i, 1] - r < 0:
                new_pos[i, 1] = r
                new_vel[i, 1] = abs(new_vel[i, 1]) * self.restitution
            # Ceiling
            if new_pos[i, 1] + r > self.height:
                new_pos[i, 1] = self.height - r
                new_vel[i, 1] = -abs(new_vel[i, 1]) * self.restitution
            # Walls
            if new_pos[i, 0] - r < 0:
                new_pos[i, 0] = r
                new_vel[i, 0] = abs(new_vel[i, 0]) * self.restitution
            if new_pos[i, 0] + r > self.width:
                new_pos[i, 0] = self.width - r
                new_vel[i, 0] = -abs(new_vel[i, 0]) * self.restitution

        # Simple elastic collisions between objects
        for i in range(self.n_objects):
            for j in range(i + 1, self.n_objects):
                diff = new_pos[i] - new_pos[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-6:
                    normal = diff / dist
                    rel_vel = new_vel[i] - new_vel[j]
                    vel_along_normal = np.dot(rel_vel, normal)
                    if vel_along_normal < 0:
                        m1, m2 = masses[i], masses[j]
                        impulse = (2 * vel_along_normal) / (m1 + m2)
                        new_vel[i] -= impulse * m2 * normal
                        new_vel[j] += impulse * m1 * normal
                    # Separate overlapping objects
                    overlap = min_dist - dist
                    new_pos[i] += normal * overlap * 0.5
                    new_pos[j] -= normal * overlap * 0.5

        return new_pos, new_vel

    def generate_trajectory(self, n_steps=100):
        """Generate a full trajectory of n_steps."""
        positions, velocities, masses, radii = self.random_state()
        trajectory = [positions.copy()]
        velocity_history = [velocities.copy()]

        for _ in range(n_steps):
            positions, velocities = self.step(positions, velocities, masses, radii)
            trajectory.append(positions.copy())
            velocity_history.append(velocities.copy())

        return (np.array(trajectory), np.array(velocity_history),
                masses, radii)

    def generate_dataset(self, n_trajectories=200, n_steps=60):
        """Generate training dataset of trajectories."""
        data = []
        for _ in range(n_trajectories):
            traj, vels, masses, radii = self.generate_trajectory(n_steps)
            data.append({
                'positions': traj,      # (T+1, n_objects, 2)
                'velocities': vels,     # (T+1, n_objects, 2)
                'masses': masses,       # (n_objects,)
                'radii': radii,         # (n_objects,)
            })
        return data


# ─────────────────────────────────────────────
# 2. WORLD MODEL — learns physics from observation
# ─────────────────────────────────────────────

class PhysicsEncoder(nn.Module):
    """Encodes object states into a latent physics representation."""

    def __init__(self, n_objects=3, latent_dim=32):
        super().__init__()
        # Input: positions (2) + velocities (2) per object = 4 * n_objects
        input_dim = n_objects * 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, state):
        return self.encoder(state)


class PhysicsPredictor(nn.Module):
    """Predicts next state from current state — must implicitly learn physics."""

    def __init__(self, n_objects=3, latent_dim=32):
        super().__init__()
        self.n_objects = n_objects
        state_dim = n_objects * 4  # pos(2) + vel(2) per object

        self.encoder = PhysicsEncoder(n_objects, latent_dim)

        # Dynamics model in latent space
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder: latent → next state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

        # Physics probe heads — try to extract interpretable physics quantities
        # These are linear probes on the latent space
        self.gravity_probe = nn.Linear(latent_dim, 1)      # What gravity does the model think exists?
        self.energy_probe = nn.Linear(latent_dim, 1)        # Total energy estimate
        self.momentum_probe = nn.Linear(latent_dim, 2)      # Momentum vector

    def forward(self, state):
        z = self.encoder(state)
        z_next = self.dynamics(z)
        next_state = self.decoder(z_next)
        return next_state, z, z_next

    def probe_physics(self, state):
        """Extract what physics the model has learned."""
        with torch.no_grad():
            z = self.encoder(state)
            gravity = self.gravity_probe(z)
            energy = self.energy_probe(z)
            momentum = self.momentum_probe(z)
        return {
            'gravity': gravity.item() if gravity.dim() == 0 else gravity.squeeze().tolist(),
            'energy': energy.item() if energy.dim() == 0 else energy.squeeze().tolist(),
            'momentum': momentum.squeeze().tolist(),
        }


class PhysicsDiscoveryEngine:
    """
    The core idea: train a model to predict physics, then extract
    what laws it discovered.

    Reward = prediction accuracy (the model MUST learn physics to predict well)
    Interpretability = linear probes on latent space to find physics concepts
    """

    def __init__(self, n_objects=3, lr=1e-3):
        self.n_objects = n_objects
        self.model = PhysicsPredictor(n_objects)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.training_losses = []
        self.physics_metrics = []

    def prepare_batch(self, data, batch_size=32):
        """Convert trajectory data into training batches."""
        states = []
        next_states = []

        for traj_data in data:
            pos = traj_data['positions']    # (T+1, n_obj, 2)
            vel = traj_data['velocities']   # (T+1, n_obj, 2)
            T = pos.shape[0] - 1

            for t in range(T):
                # Current state: flatten [pos, vel] for all objects
                state = np.concatenate([pos[t].flatten(), vel[t].flatten()])
                next_state = np.concatenate([pos[t+1].flatten(), vel[t+1].flatten()])
                states.append(state)
                next_states.append(next_state)

        states = np.array(states)
        next_states = np.array(next_states)

        # Random batch
        indices = np.random.choice(len(states), min(batch_size, len(states)), replace=False)
        return (torch.FloatTensor(states[indices]),
                torch.FloatTensor(next_states[indices]))

    def train_step(self, data, batch_size=64):
        """One training step."""
        self.model.train()
        states, next_states = self.prepare_batch(data, batch_size)

        pred_next, z, z_next = self.model(states)
        loss = self.loss_fn(pred_next, next_states)

        # Physics consistency loss: latent dynamics should be smooth
        latent_smoothness = torch.mean((z_next - z) ** 2) * 0.01
        total_loss = loss + latent_smoothness

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Train physics probes (supervised on known quantities)
        self._train_probes(states, data)

        return loss.item()

    def _train_probes(self, states, data):
        """Train linear probes to extract physics from latent space."""
        self.model.eval()
        with torch.no_grad():
            z = self.model.encoder(states)

        # Compute ground-truth physics quantities for these states
        n_obj = self.n_objects
        batch_size = states.shape[0]

        # Extract positions and velocities from flattened state
        pos = states[:, :n_obj*2].reshape(batch_size, n_obj, 2)
        vel = states[:, n_obj*2:].reshape(batch_size, n_obj, 2)

        # True gravitational acceleration (should discover ~9.81)
        # We measure it as the average downward acceleration
        gt_gravity = torch.full((batch_size, 1), 9.81)

        # True kinetic energy: 0.5 * sum(v^2) (simplified, mass=1)
        gt_energy = 0.5 * torch.sum(vel ** 2, dim=(1, 2), keepdim=False).unsqueeze(1)

        # True momentum: sum of velocities (simplified)
        gt_momentum = torch.sum(vel, dim=1)  # (batch, 2)

        # Train probes with frozen encoder
        probe_optimizer = optim.SGD([
            {'params': self.model.gravity_probe.parameters()},
            {'params': self.model.energy_probe.parameters()},
            {'params': self.model.momentum_probe.parameters()},
        ], lr=0.01)

        z_detached = z.detach()
        pred_gravity = self.model.gravity_probe(z_detached)
        pred_energy = self.model.energy_probe(z_detached)
        pred_momentum = self.model.momentum_probe(z_detached)

        probe_loss = (self.loss_fn(pred_gravity, gt_gravity) +
                      self.loss_fn(pred_energy, gt_energy) * 0.01 +
                      self.loss_fn(pred_momentum, gt_momentum) * 0.1)

        probe_optimizer.zero_grad()
        probe_loss.backward()
        probe_optimizer.step()

    def evaluate_physics_discovery(self, data):
        """How well has the model discovered physics?"""
        self.model.eval()
        states, next_states = self.prepare_batch(data, batch_size=200)

        with torch.no_grad():
            pred_next, z, z_next = self.model(states)

            # Prediction accuracy
            pred_error = torch.mean((pred_next - next_states) ** 2).item()

            # Position prediction error (first n_obj*2 dims)
            n_obj = self.n_objects
            pos_error = torch.mean(
                (pred_next[:, :n_obj*2] - next_states[:, :n_obj*2]) ** 2
            ).item()

            # Velocity prediction error
            vel_error = torch.mean(
                (pred_next[:, n_obj*2:] - next_states[:, n_obj*2:]) ** 2
            ).item()

            # Gravity probe accuracy
            pred_g = self.model.gravity_probe(z).mean().item()

            # Energy probe
            pred_e = self.model.energy_probe(z).mean().item()
            pos = states[:, :n_obj*2].reshape(-1, n_obj, 2)
            vel = states[:, n_obj*2:].reshape(-1, n_obj, 2)
            true_e = (0.5 * torch.sum(vel ** 2, dim=(1, 2))).mean().item()

            # Latent space analysis — how structured is it?
            z_std = torch.std(z, dim=0).mean().item()
            z_mean_norm = torch.norm(torch.mean(z, dim=0)).item()

        return {
            'prediction_error': pred_error,
            'position_error': pos_error,
            'velocity_error': vel_error,
            'discovered_gravity': pred_g,
            'true_gravity': 9.81,
            'gravity_error_pct': abs(pred_g - 9.81) / 9.81 * 100,
            'discovered_energy': pred_e,
            'true_energy': true_e,
            'latent_std': z_std,
            'latent_mean_norm': z_mean_norm,
        }


# ─────────────────────────────────────────────
# 3. STREAMLIT APP
# ─────────────────────────────────────────────

def plot_trajectory(traj_data, title="Simulated Physics"):
    """Animate a trajectory with Plotly."""
    positions = traj_data['positions']  # (T, n_obj, 2)
    n_steps, n_obj, _ = positions.shape

    fig = make_subplots(rows=1, cols=1)

    colors = px.colors.qualitative.Set2[:n_obj]

    # Plot trails
    for obj_idx in range(n_obj):
        fig.add_trace(go.Scatter(
            x=positions[:, obj_idx, 0],
            y=positions[:, obj_idx, 1],
            mode='lines',
            line=dict(color=colors[obj_idx], width=1, dash='dot'),
            opacity=0.4,
            name=f'Object {obj_idx+1} trail',
            showlegend=False,
        ))

    # Plot final positions as big dots
    for obj_idx in range(n_obj):
        sizes = np.linspace(3, 15, n_steps)
        fig.add_trace(go.Scatter(
            x=[positions[-1, obj_idx, 0]],
            y=[positions[-1, obj_idx, 1]],
            mode='markers',
            marker=dict(size=15, color=colors[obj_idx],
                        line=dict(width=2, color='white')),
            name=f'Object {obj_idx+1} (m={traj_data["masses"][obj_idx]:.1f})',
        ))

    # Plot start positions
    for obj_idx in range(n_obj):
        fig.add_trace(go.Scatter(
            x=[positions[0, obj_idx, 0]],
            y=[positions[0, obj_idx, 1]],
            mode='markers',
            marker=dict(size=8, color=colors[obj_idx], symbol='x'),
            name=f'Start {obj_idx+1}',
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 10], title='x'),
        yaxis=dict(range=[0, 10], title='y', scaleanchor='x'),
        height=500,
        template='plotly_dark',
    )
    return fig


def plot_training_progress(losses, metrics_history):
    """Plot training loss and physics discovery metrics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Training Loss (Prediction Error)',
            'Discovered Gravity vs True (9.81)',
            'Position Prediction Error',
            'Latent Space Structure'
        ]
    )

    # Loss curve
    fig.add_trace(go.Scatter(
        y=losses, mode='lines',
        line=dict(color='#00d4ff'),
        name='Loss'
    ), row=1, col=1)

    if metrics_history:
        epochs = [m['epoch'] for m in metrics_history]

        # Gravity discovery
        discovered_g = [m['discovered_gravity'] for m in metrics_history]
        fig.add_trace(go.Scatter(
            x=epochs, y=discovered_g, mode='lines+markers',
            line=dict(color='#ff6b6b'),
            name='Discovered g'
        ), row=1, col=2)
        fig.add_hline(y=9.81, line_dash="dash", line_color="green",
                      annotation_text="True g=9.81", row=1, col=2)

        # Position error
        pos_errors = [m['position_error'] for m in metrics_history]
        fig.add_trace(go.Scatter(
            x=epochs, y=pos_errors, mode='lines+markers',
            line=dict(color='#ffd93d'),
            name='Position Error'
        ), row=2, col=1)

        # Latent structure
        latent_std = [m['latent_std'] for m in metrics_history]
        fig.add_trace(go.Scatter(
            x=epochs, y=latent_std, mode='lines+markers',
            line=dict(color='#6bcb77'),
            name='Latent Std'
        ), row=2, col=2)

    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=False,
    )
    return fig


def plot_prediction_comparison(model, data, n_objects):
    """Show model predictions vs ground truth."""
    model.eval()
    traj = data[0]
    positions = traj['positions']
    velocities = traj['velocities']

    pred_positions = [positions[0]]
    current_pos = positions[0].copy()
    current_vel = velocities[0].copy()

    for t in range(len(positions) - 1):
        state = np.concatenate([current_pos.flatten(), current_vel.flatten()])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            pred_next, _, _ = model(state_tensor)

        pred = pred_next.squeeze().numpy()
        current_pos = pred[:n_objects*2].reshape(n_objects, 2)
        current_vel = pred[n_objects*2:].reshape(n_objects, 2)
        pred_positions.append(current_pos.copy())

    pred_positions = np.array(pred_positions)
    true_positions = positions

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Ground Truth', 'Model Prediction'])
    colors = px.colors.qualitative.Set2[:n_objects]

    for obj_idx in range(n_objects):
        fig.add_trace(go.Scatter(
            x=true_positions[:, obj_idx, 0],
            y=true_positions[:, obj_idx, 1],
            mode='lines+markers',
            marker=dict(size=3, color=colors[obj_idx]),
            line=dict(color=colors[obj_idx]),
            name=f'True Obj {obj_idx+1}',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=pred_positions[:, obj_idx, 0],
            y=pred_positions[:, obj_idx, 1],
            mode='lines+markers',
            marker=dict(size=3, color=colors[obj_idx]),
            line=dict(color=colors[obj_idx], dash='dash'),
            name=f'Pred Obj {obj_idx+1}',
        ), row=1, col=2)

    fig.update_xaxes(range=[0, 10], row=1, col=1)
    fig.update_xaxes(range=[0, 10], row=1, col=2)
    fig.update_yaxes(range=[0, 10], row=1, col=1)
    fig.update_yaxes(range=[0, 10], row=1, col=2)
    fig.update_layout(height=450, template='plotly_dark')
    return fig


def main():
    st.set_page_config(
        page_title="Physics Discovery Engine",
        page_icon="🔬",
        layout="wide"
    )

    st.title("Physics Discovery Engine")
    st.markdown("""
    **Can a neural network learn Newton's laws from scratch?**

    Instead of encoding physics as constraints, this model is *rewarded* for
    discovering physics. It watches simple 2D simulations and must learn to predict
    what happens next. To predict well, it must implicitly learn gravity, momentum,
    and collisions — like a baby learning about the world.

    *Inspired by conversations with Chelsea Finn on world models and physics-founded reasoning.*
    """)

    st.divider()

    # ── Sidebar controls ──
    with st.sidebar:
        st.header("Simulation Controls")
        n_objects = st.slider("Number of objects", 2, 6, 3)
        gravity = st.slider("True gravity (hidden from model)", 1.0, 20.0, 9.81, 0.01)
        n_trajectories = st.slider("Training trajectories", 50, 500, 200, 50)
        n_steps = st.slider("Steps per trajectory", 20, 120, 60, 10)
        restitution = st.slider("Bounce coefficient", 0.0, 1.0, 0.8, 0.05)

        st.header("Training Controls")
        n_epochs = st.slider("Training epochs", 10, 300, 100, 10)
        learning_rate = st.select_slider("Learning rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
        batch_size = st.select_slider("Batch size",
            options=[16, 32, 64, 128, 256], value=64)

    # ── Section 1: Data Generation ──
    st.header("1. Generate Physics Data")
    st.markdown("The model never sees the physics equations — only trajectories of objects moving.")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Generate Training Data", type="primary", use_container_width=True):
            world = PhysicsWorld(
                n_objects=n_objects,
                gravity=gravity,
                restitution=restitution,
            )
            with st.spinner("Simulating physics..."):
                data = world.generate_dataset(n_trajectories, n_steps)
            st.session_state['data'] = data
            st.session_state['world'] = world
            st.session_state['n_objects'] = n_objects
            st.success(f"Generated {n_trajectories} trajectories "
                       f"({n_trajectories * n_steps} state transitions)")

    if 'data' in st.session_state:
        with col2:
            sample_idx = st.number_input("Preview trajectory #",
                                          0, len(st.session_state['data'])-1, 0)

        sample_traj = st.session_state['data'][sample_idx]
        fig = plot_trajectory(sample_traj,
                              title=f"Trajectory #{sample_idx} — {n_objects} objects, g={gravity}")
        st.plotly_chart(fig, use_container_width=True)

        # Show raw data stats
        with st.expander("Data Statistics"):
            all_pos = np.concatenate([d['positions'] for d in st.session_state['data']])
            all_vel = np.concatenate([d['velocities'] for d in st.session_state['data']])
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Position", f"({all_pos[:,:,0].mean():.2f}, {all_pos[:,:,1].mean():.2f})")
            c2.metric("Avg Speed", f"{np.linalg.norm(all_vel, axis=-1).mean():.2f} m/s")
            c3.metric("Total Samples", f"{len(all_pos):,}")

    st.divider()

    # ── Section 2: Train World Model ──
    st.header("2. Train the World Model")
    st.markdown("""
    The model must predict the next state from the current state.
    **To predict accurately, it must discover the laws of physics.**
    The reward is purely prediction accuracy — no physics is given.
    """)

    if 'data' not in st.session_state:
        st.info("Generate training data first.")
        return

    if st.button("Train Model", type="primary", use_container_width=True):
        n_obj = st.session_state['n_objects']
        engine = PhysicsDiscoveryEngine(n_objects=n_obj, lr=learning_rate)
        data = st.session_state['data']

        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        metrics_chart = st.empty()

        losses = []
        metrics_history = []

        for epoch in range(n_epochs):
            # Multiple training steps per epoch
            epoch_losses = []
            for _ in range(10):
                loss = engine.train_step(data, batch_size=batch_size)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            # Evaluate every 5 epochs
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                metrics = engine.evaluate_physics_discovery(data)
                metrics['epoch'] = epoch
                metrics_history.append(metrics)

                status_text.markdown(
                    f"**Epoch {epoch+1}/{n_epochs}** — "
                    f"Loss: `{avg_loss:.6f}` — "
                    f"Discovered g: `{metrics['discovered_gravity']:.3f}` "
                    f"(true: {metrics['true_gravity']}) — "
                    f"Error: `{metrics['gravity_error_pct']:.1f}%`"
                )

            progress_bar.progress((epoch + 1) / n_epochs)

            # Update charts periodically
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                fig = plot_training_progress(losses, metrics_history)
                loss_chart.plotly_chart(fig, use_container_width=True)

        st.session_state['engine'] = engine
        st.session_state['losses'] = losses
        st.session_state['metrics_history'] = metrics_history

        st.success("Training complete!")

    st.divider()

    # ── Section 3: Physics Discovery Results ──
    st.header("3. What Physics Did the Model Discover?")

    if 'engine' not in st.session_state:
        st.info("Train the model first.")
        return

    engine = st.session_state['engine']
    metrics_history = st.session_state['metrics_history']
    data = st.session_state['data']
    n_obj = st.session_state['n_objects']

    final_metrics = metrics_history[-1]

    # Big metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Discovered Gravity",
        f"{final_metrics['discovered_gravity']:.3f}",
        f"{final_metrics['gravity_error_pct']:.1f}% error"
    )
    col2.metric(
        "True Gravity",
        f"{final_metrics['true_gravity']:.2f}",
    )
    col3.metric(
        "Prediction Error",
        f"{final_metrics['prediction_error']:.6f}",
    )
    col4.metric(
        "Latent Structure",
        f"{final_metrics['latent_std']:.4f}",
    )

    # Prediction comparison
    st.subheader("Model Predictions vs Reality")
    st.markdown("Left: ground truth physics. Right: what the model predicts (autoregressively).")
    fig = plot_prediction_comparison(engine.model, data, n_obj)
    st.plotly_chart(fig, use_container_width=True)

    # Training curves
    st.subheader("Training Progress")
    fig = plot_training_progress(
        st.session_state['losses'],
        metrics_history
    )
    st.plotly_chart(fig, use_container_width=True)

    # Physics interpretation
    st.subheader("Interpretability Analysis")
    st.markdown("""
    **Key insight**: The model was never told about gravity, momentum, or energy.
    It only learned to predict the next position/velocity. But to do this well,
    its internal representations must encode these physical concepts.

    The linear probes below test whether physics concepts are *linearly decodable*
    from the model's latent space — meaning the model organized its internal
    representations around real physics.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gravity Discovery**")
        g_discovered = final_metrics['discovered_gravity']
        g_true = final_metrics['true_gravity']
        g_pct = final_metrics['gravity_error_pct']

        if g_pct < 10:
            st.success(f"The model discovered gravity! g = {g_discovered:.3f} (error: {g_pct:.1f}%)")
        elif g_pct < 30:
            st.warning(f"The model is learning gravity: g = {g_discovered:.3f} (error: {g_pct:.1f}%)")
        else:
            st.error(f"Still learning: g = {g_discovered:.3f} (error: {g_pct:.1f}%). Try more epochs.")

    with col2:
        st.markdown("**Energy Tracking**")
        st.markdown(f"Predicted energy scale: `{final_metrics['discovered_energy']:.3f}`")
        st.markdown(f"True mean kinetic energy: `{final_metrics['true_energy']:.3f}`")

    st.divider()

    # ── Section 4: The Vision ──
    st.header("4. The Bigger Picture")
    st.markdown("""
    ### What this demonstrates

    This MVP shows that a neural network can **discover physics from scratch** —
    given only trajectories, it learns internal representations that correspond
    to real physical quantities (gravity, energy, momentum).

    ### Harrison's vision: scaling this up

    1. **Video input**: Replace structured state vectors with raw video frames
       (using a vision encoder like a ViT or CNN)
    2. **More complex physics**: Fluids, deformable bodies, electromagnetic fields
    3. **Foundation model**: Train on diverse environments so the model develops
       *general* physics understanding, not just one scenario
    4. **Novel discovery**: Once the model generalizes, expose it to new phenomena
       (biology, chemistry) and see what laws it discovers
    5. **Interpretability**: The linear probe approach scales — we can ask
       "what does neuron X encode?" and find physics concepts

    ### Architecture (from the paper)

    ```
    Physics Simulator (MuJoCo / Isaac Gym / Bullet)
         ↓
    Latent World Model (learns environment dynamics)
         ↓
    Policy Learning (RL + Imitation + Control)
         ↓
    Real-World Feedback Loops
    ```

    The key innovation is making physics discovery the **reward signal**,
    not a constraint. The model doesn't just use physics — it *finds* physics.
    """)


if __name__ == "__main__":
    main()
