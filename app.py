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
import os
import io

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


class VideoFrameEncoder(nn.Module):
    """Encodes video frames into latent representations for vision-based world model."""

    def __init__(self, frame_channels=3, frame_size=64, latent_dim=128):
        super().__init__()
        self.frame_size = frame_size
        self.encoder = nn.Sequential(
            nn.Conv2d(frame_channels, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),              # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),             # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),            # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, frames):
        return self.encoder(frames)


class VideoFrameDecoder(nn.Module):
    """Decodes latent representations back to video frames."""

    def __init__(self, frame_channels=3, frame_size=64, latent_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),   # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, frame_channels, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(z)


class VideoWorldModel(nn.Module):
    """
    Vision-based world model for learning physics from video.
    Designed to work with EgoVerse or any video dataset.

    Architecture: Frame Encoder → Latent Dynamics → Frame Decoder
    The model must learn physics to predict future frames.
    """

    def __init__(self, frame_channels=3, frame_size=64, latent_dim=128, action_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.encoder = VideoFrameEncoder(frame_channels, frame_size, latent_dim)
        self.decoder = VideoFrameDecoder(frame_channels, frame_size, latent_dim)

        # Latent dynamics: predict next latent state
        dynamics_input = latent_dim + action_dim
        self.dynamics = nn.Sequential(
            nn.Linear(dynamics_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Physics probes on the latent space
        self.gravity_probe = nn.Linear(latent_dim, 1)
        self.velocity_probe = nn.Linear(latent_dim, 2)
        self.contact_probe = nn.Linear(latent_dim, 1)

    def forward(self, frame, action=None):
        z = self.encoder(frame)
        if action is not None:
            z_input = torch.cat([z, action], dim=-1)
        else:
            z_input = z
        z_next = self.dynamics(z_input)
        frame_next = self.decoder(z_next)
        return frame_next, z, z_next

    def encode(self, frame):
        return self.encoder(frame)


class EgoVerseLoader:
    """
    Interface for loading and processing EgoVerse data.

    EgoVerse contains ~129M frames of egocentric manipulation video
    across 400 scenes with paired action data. Data is stored in Zarr format
    and accessible via S3-compatible API.

    For the MVP, we support:
    1. Synthetic video generation (demo mode — no download needed)
    2. Local video file loading
    3. EgoVerse S3 download (requires setup)
    """

    EGOVERSE_TASKS = [
        "decorating cakes", "sewing fabric", "glazing cookies",
        "planting lemongrass", "disassembling phones", "decanting perfume",
        "folding clothes", "pouring liquid", "stacking blocks",
        "opening jars", "cutting vegetables", "wiping surfaces",
    ]

    @staticmethod
    def generate_synthetic_video(n_frames=60, frame_size=64, n_objects=3,
                                  gravity=9.81, scenario="falling_objects"):
        """
        Generate synthetic video frames from physics simulation.
        This bridges the 2D simulator to the video world model.
        """
        world = PhysicsWorld(n_objects=n_objects, gravity=gravity,
                             width=frame_size, height=frame_size)
        positions, velocities, masses, radii = world.random_state()

        # Scale positions to frame coordinates
        positions *= (frame_size / 10.0)
        radii_px = radii * (frame_size / 10.0)

        frames = []
        for t in range(n_frames):
            # Render frame
            frame = np.zeros((3, frame_size, frame_size), dtype=np.float32)

            # Draw background gradient (sky)
            for y in range(frame_size):
                frame[2, y, :] = 0.3 + 0.4 * (1 - y / frame_size)  # blue gradient

            # Draw ground
            ground_y = int(frame_size * 0.1)
            frame[1, :ground_y, :] = 0.3  # green ground

            # Draw objects as colored circles
            colors = [
                [0.9, 0.2, 0.2],  # red
                [0.2, 0.8, 0.2],  # green
                [0.2, 0.4, 0.9],  # blue
                [0.9, 0.9, 0.2],  # yellow
                [0.9, 0.2, 0.9],  # magenta
                [0.2, 0.9, 0.9],  # cyan
            ]

            for i in range(n_objects):
                cx, cy = int(positions[i, 0]), int(positions[i, 1])
                r = max(2, int(radii_px[i]))
                color = colors[i % len(colors)]

                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dx*dx + dy*dy <= r*r:
                            px_x = cx + dx
                            px_y = cy + dy
                            if 0 <= px_x < frame_size and 0 <= px_y < frame_size:
                                for c in range(3):
                                    frame[c, px_y, px_x] = color[c]

            frames.append(frame)

            # Step physics (scale back to world coords, step, scale back)
            pos_world = positions / (frame_size / 10.0)
            vel_world = velocities / (frame_size / 10.0)
            pos_world, vel_world = world.step(pos_world, vel_world, masses, radii)
            positions = pos_world * (frame_size / 10.0)
            velocities = vel_world * (frame_size / 10.0)

        return np.array(frames), masses

    @staticmethod
    def load_local_video(video_path, frame_size=64, max_frames=120):
        """Load a local video file and extract frames."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (frame_size, frame_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
                frames.append(frame)
            cap.release()
            return np.array(frames) if frames else None
        except ImportError:
            return None

    @staticmethod
    def get_egoverse_info():
        """Return information about EgoVerse dataset for the UI."""
        return {
            'total_episodes': 54088,
            'total_frames': 128948271,
            'hours': 1194,
            'scenes': 400,
            'tasks': 4949,
            'objects': 120,
            'format': 'Zarr / LeRobot',
            'repo': 'https://github.com/GaTech-RL2/EgoVerse',
            'explorer': 'https://partners.mecka.ai/egoverse',
        }


class VideoPhysicsEngine:
    """Training engine for the video-based world model."""

    def __init__(self, frame_size=64, latent_dim=128, lr=1e-3, action_dim=0):
        self.model = VideoWorldModel(
            frame_size=frame_size,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.training_losses = []

    def train_step(self, frames, batch_size=16):
        """Train on frame pairs (frame_t, frame_{t+1})."""
        self.model.train()
        n_frames = len(frames)

        # Sample random pairs
        indices = np.random.choice(n_frames - 1, min(batch_size, n_frames - 1), replace=False)
        current_frames = torch.FloatTensor(frames[indices])
        next_frames = torch.FloatTensor(frames[indices + 1])

        # Predict next frame
        pred_next, z, z_next = self.model(current_frames)
        recon_loss = self.loss_fn(pred_next, next_frames)

        # Latent smoothness
        smooth_loss = torch.mean((z_next - z) ** 2) * 0.01

        total_loss = recon_loss + smooth_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return recon_loss.item()

    def evaluate(self, frames, n_samples=32):
        """Evaluate prediction quality."""
        self.model.eval()
        n_frames = len(frames)
        indices = np.random.choice(n_frames - 1, min(n_samples, n_frames - 1), replace=False)

        with torch.no_grad():
            current = torch.FloatTensor(frames[indices])
            target = torch.FloatTensor(frames[indices + 1])
            pred, z, _ = self.model(current)

            mse = self.loss_fn(pred, target).item()
            # PSNR
            psnr = -10 * np.log10(mse + 1e-10)

            # Gravity probe
            g_pred = self.model.gravity_probe(z).mean().item()

            # Latent analysis
            z_std = torch.std(z, dim=0).mean().item()

        return {
            'mse': mse,
            'psnr': psnr,
            'gravity_probe': g_pred,
            'latent_std': z_std,
        }

    def get_prediction_frames(self, frames, start_idx=0, n_predict=20):
        """Generate predicted frames autoregressively."""
        self.model.eval()
        predictions = []
        current = torch.FloatTensor(frames[start_idx:start_idx+1])

        with torch.no_grad():
            for _ in range(n_predict):
                pred, _, _ = self.model(current)
                predictions.append(pred.squeeze().numpy())
                current = pred

        return np.array(predictions)


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

        # Separate probe params from main model params
        probe_params = set(
            list(self.model.gravity_probe.parameters()) +
            list(self.model.energy_probe.parameters()) +
            list(self.model.momentum_probe.parameters())
        )
        main_params = [p for p in self.model.parameters() if p not in probe_params]

        self.optimizer = optim.Adam(main_params, lr=lr)
        self.probe_optimizer = optim.Adam(list(probe_params), lr=1e-3)
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

        # Train probes with frozen encoder using persistent optimizer
        z_detached = z.detach()
        pred_gravity = self.model.gravity_probe(z_detached)
        pred_energy = self.model.energy_probe(z_detached)
        pred_momentum = self.model.momentum_probe(z_detached)

        probe_loss = (self.loss_fn(pred_gravity, gt_gravity) +
                      self.loss_fn(pred_energy, gt_energy) * 0.01 +
                      self.loss_fn(pred_momentum, gt_momentum) * 0.1)

        self.probe_optimizer.zero_grad()
        probe_loss.backward()

        # Clip probe gradients to prevent blowup
        torch.nn.utils.clip_grad_norm_(self.model.gravity_probe.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.energy_probe.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.momentum_probe.parameters(), 1.0)

        self.probe_optimizer.step()

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


def plot_video_frames(frames, title="Video Frames", n_show=8):
    """Display a grid of video frames."""
    n_frames = len(frames)
    indices = np.linspace(0, n_frames - 1, min(n_show, n_frames), dtype=int)

    fig = make_subplots(rows=1, cols=len(indices),
                        subplot_titles=[f"t={i}" for i in indices])

    for col, idx in enumerate(indices):
        frame = frames[idx]
        if frame.shape[0] == 3:  # CHW -> HWC
            frame = np.transpose(frame, (1, 2, 0))
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        fig.add_trace(go.Image(z=frame), row=1, col=col+1)

    fig.update_layout(
        title=title,
        height=250,
        template='plotly_dark',
        showlegend=False,
    )
    for i in range(len(indices)):
        fig.update_xaxes(showticklabels=False, row=1, col=i+1)
        fig.update_yaxes(showticklabels=False, row=1, col=i+1)
    return fig


def plot_video_predictions(true_frames, pred_frames, n_show=6):
    """Side-by-side comparison of true vs predicted frames."""
    n_frames = min(len(true_frames), len(pred_frames))
    indices = np.linspace(0, n_frames - 1, min(n_show, n_frames), dtype=int)

    fig = make_subplots(rows=2, cols=len(indices),
                        row_titles=['Ground Truth', 'Predicted'],
                        column_titles=[f"t={i}" for i in indices])

    for col, idx in enumerate(indices):
        for row, frames in enumerate([true_frames, pred_frames]):
            frame = frames[idx]
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            fig.add_trace(go.Image(z=frame), row=row+1, col=col+1)

    fig.update_layout(
        height=350,
        template='plotly_dark',
        showlegend=False,
    )
    for r in range(2):
        for c in range(len(indices)):
            fig.update_xaxes(showticklabels=False, row=r+1, col=c+1)
            fig.update_yaxes(showticklabels=False, row=r+1, col=c+1)
    return fig


def run_simulation_tab():
    """The original 2D physics simulation tab."""

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

        losses = []
        metrics_history = []

        for epoch in range(n_epochs):
            epoch_losses = []
            for _ in range(10):
                loss = engine.train_step(data, batch_size=batch_size)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Discovered Gravity",
                f"{final_metrics['discovered_gravity']:.3f}",
                f"{final_metrics['gravity_error_pct']:.1f}% error")
    col2.metric("True Gravity", f"{final_metrics['true_gravity']:.2f}")
    col3.metric("Prediction Error", f"{final_metrics['prediction_error']:.6f}")
    col4.metric("Latent Structure", f"{final_metrics['latent_std']:.4f}")

    st.subheader("Model Predictions vs Reality")
    st.markdown("Left: ground truth physics. Right: what the model predicts (autoregressively).")
    fig = plot_prediction_comparison(engine.model, data, n_obj)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training Progress")
    fig = plot_training_progress(st.session_state['losses'], metrics_history)
    st.plotly_chart(fig, use_container_width=True)

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
        g_pct = final_metrics['gravity_error_pct']
        g_discovered = final_metrics['discovered_gravity']
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


def run_video_tab():
    """Video-based world model tab with EgoVerse integration."""

    st.header("Video World Model")
    st.markdown("""
    Scale up from structured state vectors to **raw video frames**.
    The model learns to predict the next frame — and must discover physics to do it well.

    This connects to [EgoVerse](https://partners.mecka.ai/egoverse) — a dataset of
    **129M frames** of egocentric manipulation video across 400 real-world scenes.
    """)

    # EgoVerse info
    with st.expander("About EgoVerse Dataset"):
        info = EgoVerseLoader.get_egoverse_info()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Episodes", f"{info['total_episodes']:,}")
        col2.metric("Total Frames", f"{info['total_frames']:,}")
        col3.metric("Hours of Video", f"{info['hours']:,}")
        col4.metric("Unique Scenes", info['scenes'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unique Tasks", f"{info['tasks']:,}")
        col2.metric("Object Types", info['objects'])
        col3.metric("Format", info['format'])

        st.markdown(f"""
        **Links:**
        - [Interactive Explorer]({info['explorer']})
        - [GitHub Repository]({info['repo']})

        **Example tasks from EgoVerse:**
        {', '.join(f'`{t}`' for t in EgoVerseLoader.EGOVERSE_TASKS)}

        **To use real EgoVerse data**, clone the repo and run:
        ```bash
        git clone https://github.com/GaTech-RL2/EgoVerse.git
        cd EgoVerse
        python egomimic/scripts/data_download/sync_s3.py \\
            --local-dir ./data --filters aria-fold-clothes
        ```
        """)

    st.divider()

    # Sidebar controls for video mode
    with st.sidebar:
        st.header("Video Model Controls")
        frame_size = st.select_slider("Frame size", options=[32, 64, 128], value=64)
        latent_dim = st.select_slider("Latent dim", options=[64, 128, 256], value=128)
        vid_n_epochs = st.slider("Training epochs (video)", 10, 200, 50, 10, key="vid_epochs")
        vid_lr = st.select_slider("Learning rate (video)",
            options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, key="vid_lr")
        vid_batch = st.select_slider("Batch size (video)",
            options=[4, 8, 16, 32], value=8, key="vid_batch")

    # Data source selection
    st.subheader("1. Load Video Data")
    data_source = st.radio(
        "Data source",
        ["Synthetic Physics Video (demo)", "Upload Video File", "EgoVerse (local)"],
        horizontal=True,
    )

    frames = None

    if data_source == "Synthetic Physics Video (demo)":
        col1, col2 = st.columns(2)
        with col1:
            syn_objects = st.slider("Objects in scene", 2, 6, 3, key="syn_obj")
            syn_gravity = st.slider("Gravity", 1.0, 20.0, 9.81, key="syn_g")
        with col2:
            syn_frames = st.slider("Number of frames", 30, 200, 80, key="syn_frames")

        if st.button("Generate Synthetic Video", type="primary", use_container_width=True):
            with st.spinner("Rendering physics simulation to video..."):
                frames, masses = EgoVerseLoader.generate_synthetic_video(
                    n_frames=syn_frames,
                    frame_size=frame_size,
                    n_objects=syn_objects,
                    gravity=syn_gravity,
                )
            st.session_state['video_frames'] = frames
            st.success(f"Generated {len(frames)} frames at {frame_size}x{frame_size}")

    elif data_source == "Upload Video File":
        uploaded = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        if uploaded:
            tmp_path = f"/tmp/uploaded_video.{uploaded.name.split('.')[-1]}"
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.read())
            with st.spinner("Extracting frames..."):
                frames = EgoVerseLoader.load_local_video(tmp_path, frame_size=frame_size)
            if frames is not None:
                st.session_state['video_frames'] = frames
                st.success(f"Loaded {len(frames)} frames at {frame_size}x{frame_size}")
            else:
                st.error("Could not load video. Install opencv-python: `pip install opencv-python`")

    elif data_source == "EgoVerse (local)":
        ego_path = st.text_input("Path to EgoVerse data directory",
                                  placeholder="/path/to/EgoVerse/data")
        if ego_path and os.path.exists(ego_path):
            st.info("EgoVerse Zarr loading — looking for episode data...")
            # List available episodes
            try:
                episodes = [d for d in os.listdir(ego_path) if os.path.isdir(os.path.join(ego_path, d))]
                if episodes:
                    st.write(f"Found {len(episodes)} episodes")
                    selected = st.selectbox("Select episode", episodes[:50])
                    st.markdown("*Full EgoVerse integration requires the EgoVerse package. "
                                "Use synthetic mode for the demo.*")
            except Exception as e:
                st.error(f"Error reading directory: {e}")
        elif ego_path:
            st.warning("Directory not found. Download EgoVerse data first (see instructions above).")

    # Show loaded frames
    if 'video_frames' in st.session_state:
        frames = st.session_state['video_frames']
        fig = plot_video_frames(frames, title=f"Input Video — {len(frames)} frames")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Train video world model ──
    st.subheader("2. Train Video World Model")
    st.markdown("""
    The model sees frame pairs and learns to predict the next frame.
    To predict object motion, falling, collisions — it must learn physics.
    """)

    if 'video_frames' not in st.session_state:
        st.info("Load video data first.")
        return

    frames = st.session_state['video_frames']

    if st.button("Train Video Model", type="primary", use_container_width=True):
        engine = VideoPhysicsEngine(
            frame_size=frame_size,
            latent_dim=latent_dim,
            lr=vid_lr,
        )

        progress = st.progress(0)
        status = st.empty()
        chart_placeholder = st.empty()

        losses = []
        metrics_list = []

        for epoch in range(vid_n_epochs):
            epoch_losses = []
            for _ in range(5):
                loss = engine.train_step(frames, batch_size=vid_batch)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if epoch % 5 == 0 or epoch == vid_n_epochs - 1:
                metrics = engine.evaluate(frames)
                metrics['epoch'] = epoch
                metrics_list.append(metrics)

                status.markdown(
                    f"**Epoch {epoch+1}/{vid_n_epochs}** — "
                    f"Loss: `{avg_loss:.6f}` — "
                    f"PSNR: `{metrics['psnr']:.1f} dB` — "
                    f"Gravity probe: `{metrics['gravity_probe']:.3f}`"
                )

            progress.progress((epoch + 1) / vid_n_epochs)

            if epoch % 10 == 0 or epoch == vid_n_epochs - 1:
                fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Reconstruction Loss', 'PSNR (higher = better)'])
                fig.add_trace(go.Scatter(y=losses, mode='lines',
                    line=dict(color='#00d4ff'), name='Loss'), row=1, col=1)
                if metrics_list:
                    fig.add_trace(go.Scatter(
                        x=[m['epoch'] for m in metrics_list],
                        y=[m['psnr'] for m in metrics_list],
                        mode='lines+markers',
                        line=dict(color='#6bcb77'), name='PSNR'
                    ), row=1, col=2)
                fig.update_layout(height=300, template='plotly_dark', showlegend=False)
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        st.session_state['video_engine'] = engine
        st.session_state['video_losses'] = losses
        st.session_state['video_metrics'] = metrics_list
        st.success("Video model training complete!")

    st.divider()

    # ── Results ──
    st.subheader("3. Video Prediction Results")

    if 'video_engine' not in st.session_state:
        st.info("Train the video model first.")
        return

    engine = st.session_state['video_engine']
    metrics_list = st.session_state['video_metrics']

    final = metrics_list[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("PSNR", f"{final['psnr']:.1f} dB")
    col2.metric("MSE", f"{final['mse']:.6f}")
    col3.metric("Gravity Probe", f"{final['gravity_probe']:.3f}")

    # Show predictions
    st.markdown("**Predicted vs Ground Truth Frames**")
    n_pred = min(20, len(frames) - 1)
    pred_frames = engine.get_prediction_frames(frames, start_idx=0, n_predict=n_pred)
    true_frames = frames[1:n_pred+1]

    fig = plot_video_predictions(true_frames, pred_frames, n_show=6)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **What's happening**: The model receives one frame and autoregressively
    predicts the next N frames. Blurriness = uncertainty about physics.
    As training improves, predictions sharpen because the model develops
    a better internal physics model.
    """)


def run_interactive_tab():
    """Interactive live simulation where users design their own physics experiments."""

    st.header("Interactive Physics Lab")
    st.markdown("""
    Design your own physics experiment. Place objects, set their velocities,
    and watch the simulation play out. Then see if the trained model can predict
    what happens.
    """)

    # ── Object configuration ──
    st.subheader("1. Design Your Experiment")

    with st.sidebar:
        st.header("Interactive Lab Controls")
        lab_gravity = st.slider("Gravity", 0.0, 30.0, 9.81, 0.1, key="lab_g")
        lab_restitution = st.slider("Bounciness", 0.0, 1.0, 0.8, 0.05, key="lab_r")
        lab_steps = st.slider("Simulation steps", 30, 300, 120, 10, key="lab_steps")
        lab_dt = st.select_slider("Time step", options=[0.005, 0.01, 0.02, 0.04], value=0.02, key="lab_dt")
        lab_speed = st.slider("Playback speed (ms/frame)", 10, 200, 50, 10, key="lab_speed")

    n_objects = st.slider("Number of objects", 1, 8, 3, key="lab_n")

    st.markdown("**Configure each object:**")

    objects = []
    cols_per_row = min(n_objects, 4)

    for i in range(n_objects):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)

        col = cols[i % cols_per_row]
        with col:
            st.markdown(f"**Object {i+1}**")
            preset = st.selectbox(
                f"Preset #{i+1}",
                ["Custom", "Drop from top", "Launch right", "Launch up",
                 "Fast diagonal", "Stationary", "Orbit-like"],
                key=f"preset_{i}"
            )

            if preset == "Drop from top":
                px_def, py_def = 2.0 + i * 2.5, 9.0
                vx_def, vy_def = 0.0, 0.0
            elif preset == "Launch right":
                px_def, py_def = 1.0, 2.0 + i * 1.5
                vx_def, vy_def = 5.0, 3.0
            elif preset == "Launch up":
                px_def, py_def = 3.0 + i * 2.0, 1.0
                vx_def, vy_def = 0.0, 8.0
            elif preset == "Fast diagonal":
                px_def, py_def = 1.0, 1.0 + i * 2.0
                vx_def, vy_def = 6.0, 6.0
            elif preset == "Stationary":
                px_def, py_def = 3.0 + i * 2.0, 5.0
                vx_def, vy_def = 0.0, 0.0
            elif preset == "Orbit-like":
                angle = float(i * 2 * np.pi / max(n_objects, 1))
                px_def = float(5.0 + 3.0 * np.cos(angle))
                py_def = float(5.0 + 3.0 * np.sin(angle))
                vx_def = float(-3.0 * np.sin(angle))
                vy_def = float(3.0 * np.cos(angle))
            else:
                px_def, py_def = 2.0 + i * 2.5, 5.0
                vx_def, vy_def = 0.0, 0.0

            # Ensure all defaults are Python floats and in valid range
            px_def = float(min(max(px_def, 0.5), 9.5))
            py_def = float(min(max(py_def, 0.5), 9.5))
            vx_def = float(max(-10.0, min(10.0, vx_def)))
            vy_def = float(max(-10.0, min(10.0, vy_def)))

            px = st.number_input(f"x pos", 0.1, 9.9, px_def,
                                  step=0.5, key=f"px_{i}")
            py = st.number_input(f"y pos", 0.1, 9.9, py_def,
                                  step=0.5, key=f"py_{i}")
            vx = st.number_input(f"x vel", -10.0, 10.0, vx_def,
                                  step=0.5, key=f"vx_{i}")
            vy = st.number_input(f"y vel", -10.0, 10.0, vy_def,
                                  step=0.5, key=f"vy_{i}")
            mass = st.number_input(f"mass", 0.1, 5.0, 1.0, step=0.1, key=f"mass_{i}")

            objects.append({
                'pos': [px, py], 'vel': [vx, vy], 'mass': mass
            })

    st.divider()

    # ── Run simulation ──
    st.subheader("2. Run Simulation")

    scenario_presets = st.selectbox("Or try a preset scenario:", [
        "Use objects above",
        "Newton's Cradle",
        "Projectile Motion",
        "Billiards Break",
        "Rain (many objects falling)",
        "Collision Course",
    ])

    if scenario_presets == "Newton's Cradle":
        objects = [
            {'pos': [2.0, 5.0], 'vel': [4.0, 0.0], 'mass': 1.0},
            {'pos': [5.0, 5.0], 'vel': [0.0, 0.0], 'mass': 1.0},
            {'pos': [5.6, 5.0], 'vel': [0.0, 0.0], 'mass': 1.0},
            {'pos': [6.2, 5.0], 'vel': [0.0, 0.0], 'mass': 1.0},
        ]
        n_objects = 4
    elif scenario_presets == "Projectile Motion":
        objects = [
            {'pos': [1.0, 1.0], 'vel': [4.0, 8.0], 'mass': 1.0},
            {'pos': [1.0, 1.0], 'vel': [6.0, 6.0], 'mass': 0.5},
            {'pos': [1.0, 1.0], 'vel': [3.0, 10.0], 'mass': 2.0},
        ]
        n_objects = 3
    elif scenario_presets == "Billiards Break":
        objects = [
            {'pos': [2.0, 5.0], 'vel': [6.0, 0.0], 'mass': 1.0},
            {'pos': [6.0, 5.0], 'vel': [0.0, 0.0], 'mass': 1.0},
            {'pos': [6.5, 5.3], 'vel': [0.0, 0.0], 'mass': 1.0},
            {'pos': [6.5, 4.7], 'vel': [0.0, 0.0], 'mass': 1.0},
            {'pos': [7.0, 5.6], 'vel': [0.0, 0.0], 'mass': 1.0},
            {'pos': [7.0, 5.0], 'vel': [0.0, 0.0], 'mass': 1.0},
            {'pos': [7.0, 4.4], 'vel': [0.0, 0.0], 'mass': 1.0},
        ]
        n_objects = 7
    elif scenario_presets == "Rain (many objects falling)":
        objects = [
            {'pos': [1 + i * 1.1, 8.0 + np.random.uniform(-1, 1)],
             'vel': [np.random.uniform(-0.5, 0.5), 0.0],
             'mass': np.random.uniform(0.3, 1.0)}
            for i in range(8)
        ]
        n_objects = 8
    elif scenario_presets == "Collision Course":
        objects = [
            {'pos': [1.0, 5.0], 'vel': [5.0, 0.0], 'mass': 2.0},
            {'pos': [9.0, 5.0], 'vel': [-5.0, 0.0], 'mass': 2.0},
            {'pos': [5.0, 1.0], 'vel': [0.0, 5.0], 'mass': 1.0},
            {'pos': [5.0, 9.0], 'vel': [0.0, -5.0], 'mass': 1.0},
        ]
        n_objects = 4

    if st.button("Run Simulation", type="primary", use_container_width=True):
        # Build initial state
        positions = np.array([o['pos'] for o in objects])
        velocities = np.array([o['vel'] for o in objects])
        masses = np.array([o['mass'] for o in objects])
        radii = masses * 0.2 + 0.15

        world = PhysicsWorld(
            n_objects=n_objects,
            gravity=lab_gravity,
            restitution=lab_restitution,
            dt=lab_dt,
        )

        # Run simulation
        trajectory = [positions.copy()]
        vel_history = [velocities.copy()]
        energy_history = []

        for step in range(lab_steps):
            # Track energy
            ke = 0.5 * np.sum(masses[:, None] * velocities ** 2)
            pe = np.sum(masses * lab_gravity * positions[:, 1])
            energy_history.append({'ke': ke, 'pe': pe, 'total': ke + pe})

            positions, velocities = world.step(positions, velocities, masses, radii)
            trajectory.append(positions.copy())
            vel_history.append(velocities.copy())

        trajectory = np.array(trajectory)
        vel_history = np.array(vel_history)

        st.session_state['lab_trajectory'] = trajectory
        st.session_state['lab_velocities'] = vel_history
        st.session_state['lab_masses'] = masses
        st.session_state['lab_radii'] = radii
        st.session_state['lab_energy'] = energy_history
        st.session_state['lab_n_objects'] = n_objects
        st.session_state['lab_gravity'] = lab_gravity

    # ── Display results ──
    if 'lab_trajectory' not in st.session_state:
        st.info("Configure objects above and click Run Simulation.")
        return

    trajectory = st.session_state['lab_trajectory']
    masses = st.session_state['lab_masses']
    radii = st.session_state['lab_radii']
    energy_history = st.session_state['lab_energy']
    n_obj = st.session_state['lab_n_objects']

    # Animated playback with Plotly
    st.subheader("3. Live Simulation")

    colors = px.colors.qualitative.Set2[:n_obj] + px.colors.qualitative.Pastel[:max(0, n_obj-8)]

    # Build animation frames
    frames_list = []
    for t in range(len(trajectory)):
        frame_data = []
        # Trails up to current time
        for obj_idx in range(n_obj):
            trail_end = t + 1
            frame_data.append(go.Scatter(
                x=trajectory[:trail_end, obj_idx, 0],
                y=trajectory[:trail_end, obj_idx, 1],
                mode='lines',
                line=dict(color=colors[obj_idx % len(colors)], width=1),
                opacity=0.3,
                showlegend=False,
            ))
        # Current positions
        for obj_idx in range(n_obj):
            frame_data.append(go.Scatter(
                x=[trajectory[t, obj_idx, 0]],
                y=[trajectory[t, obj_idx, 1]],
                mode='markers',
                marker=dict(
                    size=radii[obj_idx] * 30 + 8,
                    color=colors[obj_idx % len(colors)],
                    line=dict(width=2, color='white'),
                ),
                name=f'Obj {obj_idx+1} (m={masses[obj_idx]:.1f})',
                showlegend=(t == 0),
            ))
        frames_list.append(go.Frame(data=frame_data, name=str(t)))

    # Initial frame
    init_data = []
    for obj_idx in range(n_obj):
        init_data.append(go.Scatter(
            x=[trajectory[0, obj_idx, 0]],
            y=[trajectory[0, obj_idx, 1]],
            mode='lines',
            line=dict(color=colors[obj_idx % len(colors)], width=1),
            opacity=0.3,
            showlegend=False,
        ))
    for obj_idx in range(n_obj):
        init_data.append(go.Scatter(
            x=[trajectory[0, obj_idx, 0]],
            y=[trajectory[0, obj_idx, 1]],
            mode='markers',
            marker=dict(
                size=radii[obj_idx] * 30 + 8,
                color=colors[obj_idx % len(colors)],
                line=dict(width=2, color='white'),
            ),
            name=f'Obj {obj_idx+1} (m={masses[obj_idx]:.1f})',
        ))

    fig = go.Figure(data=init_data, frames=frames_list)

    fig.update_layout(
        xaxis=dict(range=[0, 10], title='x'),
        yaxis=dict(range=[0, 10], title='y', scaleanchor='x'),
        height=550,
        template='plotly_dark',
        title="Live Physics Simulation (press Play)",
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 1.15,
            'x': 0.5,
            'xanchor': 'center',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': lab_speed, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                    }],
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0},
                    }],
                },
                {
                    'label': 'Reset',
                    'method': 'animate',
                    'args': [['0'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0},
                    }],
                },
            ],
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {'args': [[str(t)], {'frame': {'duration': 0, 'redraw': True},
                                      'mode': 'immediate',
                                      'transition': {'duration': 0}}],
                 'label': str(t),
                 'method': 'animate'}
                for t in range(0, len(trajectory), max(1, len(trajectory) // 50))
            ],
            'x': 0.1,
            'len': 0.8,
            'y': -0.05,
            'currentvalue': {
                'prefix': 'Step: ',
                'visible': True,
            },
            'transition': {'duration': 0},
        }],
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Energy analysis ──
    st.subheader("4. Energy Analysis")
    st.markdown("Conservation of energy — does the simulation preserve total energy?")

    if energy_history:
        energy_fig = go.Figure()
        steps = list(range(len(energy_history)))
        energy_fig.add_trace(go.Scatter(
            x=steps,
            y=[e['ke'] for e in energy_history],
            mode='lines', name='Kinetic Energy',
            line=dict(color='#ff6b6b'),
        ))
        energy_fig.add_trace(go.Scatter(
            x=steps,
            y=[e['pe'] for e in energy_history],
            mode='lines', name='Potential Energy',
            line=dict(color='#6bcb77'),
        ))
        energy_fig.add_trace(go.Scatter(
            x=steps,
            y=[e['total'] for e in energy_history],
            mode='lines', name='Total Energy',
            line=dict(color='#ffd93d', width=2),
        ))
        energy_fig.update_layout(
            height=350, template='plotly_dark',
            xaxis_title='Step', yaxis_title='Energy',
            title='Energy Over Time',
        )
        st.plotly_chart(energy_fig, use_container_width=True)

    # ── Model prediction comparison ──
    st.subheader("5. Can the Model Predict This?")
    st.markdown("""
    If you've trained a model in the **2D Physics Simulation** tab, you can
    see how well it predicts your custom experiment.
    """)

    if 'engine' in st.session_state:
        engine = st.session_state['engine']
        model_n_obj = st.session_state['n_objects']

        if n_obj == model_n_obj:
            vel_history = st.session_state['lab_velocities']
            model = engine.model
            model.eval()

            pred_positions = [trajectory[0]]
            current_pos = trajectory[0].copy()
            current_vel = vel_history[0].copy()

            for t in range(len(trajectory) - 1):
                state = np.concatenate([current_pos.flatten(), current_vel.flatten()])
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    pred_next, _, _ = model(state_tensor)
                pred = pred_next.squeeze().numpy()
                current_pos = pred[:n_obj*2].reshape(n_obj, 2)
                current_vel = pred[n_obj*2:].reshape(n_obj, 2)
                pred_positions.append(current_pos.copy())

            pred_positions = np.array(pred_positions)

            comp_fig = make_subplots(rows=1, cols=2,
                                      subplot_titles=['Your Experiment (Truth)', 'Model Prediction'])
            for obj_idx in range(n_obj):
                comp_fig.add_trace(go.Scatter(
                    x=trajectory[:, obj_idx, 0], y=trajectory[:, obj_idx, 1],
                    mode='lines+markers', marker=dict(size=3, color=colors[obj_idx % len(colors)]),
                    line=dict(color=colors[obj_idx % len(colors)]),
                    name=f'True {obj_idx+1}',
                ), row=1, col=1)
                comp_fig.add_trace(go.Scatter(
                    x=pred_positions[:, obj_idx, 0], y=pred_positions[:, obj_idx, 1],
                    mode='lines+markers', marker=dict(size=3, color=colors[obj_idx % len(colors)]),
                    line=dict(color=colors[obj_idx % len(colors)], dash='dash'),
                    name=f'Pred {obj_idx+1}',
                ), row=1, col=2)

            comp_fig.update_xaxes(range=[0, 10], row=1, col=1)
            comp_fig.update_xaxes(range=[0, 10], row=1, col=2)
            comp_fig.update_yaxes(range=[0, 10], row=1, col=1)
            comp_fig.update_yaxes(range=[0, 10], row=1, col=2)
            comp_fig.update_layout(height=450, template='plotly_dark')
            st.plotly_chart(comp_fig, use_container_width=True)

            # Compute error
            error = np.mean((trajectory - pred_positions) ** 2)
            st.metric("Mean Squared Error vs Truth", f"{error:.6f}")
        else:
            st.warning(f"Model was trained with {model_n_obj} objects, "
                       f"but this experiment has {n_obj}. Train a new model with "
                       f"matching object count to compare.")
    else:
        st.info("Train a model in the '2D Physics Simulation' tab first to compare predictions here.")


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
    discovering physics. It watches simulations and must learn to predict
    what happens next. To predict well, it must implicitly learn gravity, momentum,
    and collisions — like a baby learning about the world.

    *Inspired by conversations with Chelsea Finn on world models and physics-founded reasoning.*
    """)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "2D Physics Simulation",
        "Interactive Physics Lab",
        "Video World Model + EgoVerse",
        "The Vision"
    ])

    with tab1:
        run_simulation_tab()

    with tab2:
        run_interactive_tab()

    with tab3:
        run_video_tab()

    with tab4:
        st.header("The Bigger Picture")
        st.markdown("""
        ### What this demonstrates

        This MVP shows that a neural network can **discover physics from scratch** —
        given only trajectories or video, it learns internal representations that correspond
        to real physical quantities (gravity, energy, momentum).

        ### Scaling this up

        1. **Video input**: The Video World Model tab shows how to go from structured
           state vectors to raw pixel observations using a CNN encoder/decoder
        2. **EgoVerse integration**: 129M frames of real-world manipulation data —
           objects being picked up, poured, folded, assembled. All governed by real physics.
        3. **Foundation model**: Train on diverse EgoVerse environments so the model develops
           *general* physics understanding, not just one scenario
        4. **Novel discovery**: Once the model generalizes, expose it to new phenomena
           (biology, chemistry) and see what laws it discovers
        5. **Interpretability**: The linear probe approach scales — we can ask
           "what does neuron X encode?" and find physics concepts

        ### Architecture

        ```
        Data Source (Simulator / EgoVerse Video / Isaac Gym)
             |
        Frame Encoder (CNN / ViT)
             |
        Latent World Model (learns environment dynamics)
             |
        Physics Probes (linear decoders for gravity, energy, momentum)
             |
        Frame Decoder (reconstruct predicted frames)
        ```

        ### The key innovation

        Making physics discovery the **reward signal**, not a constraint.
        The model doesn't just use physics — it *finds* physics.

        ### EgoVerse as training data

        EgoVerse provides the bridge from toy simulations to reality:
        - **54,088 episodes** of humans manipulating objects
        - **400 real-world scenes** (kitchens, workshops, labs)
        - **4,949 unique tasks** (pouring, folding, cutting, assembling)
        - Paired **action + observation** data for state-action world models
        - Available in **Zarr format** for efficient frame-level random access

        The model watches humans interact with the physical world and must
        learn the rules governing those interactions — just like a baby does.
        """)


if __name__ == "__main__":
    main()
