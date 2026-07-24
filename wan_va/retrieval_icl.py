# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Progress estimation for Video ICL (planning_icl.md section 2.2).

Two estimators, both monotonic (progress can only advance) and range-bounded
(search window [prev, prev + delta] to kill multi-peak jumps from repetitive
motions):

  * GTProgressMatcher — Phase 1 (E1): privileged-sim-state matching. The
    LIBERO client compares the CURRENT mujoco sim state against the demo's
    per-step `states` array (L2) and maps the matched env step to a demo
    chunk index. This decouples "does demo injection help at all" from
    retrieval quality.

  * LatentRetriever — Phase 2 (E4): cosine argmax over pooled VAE-latent
    features of the LATEST REAL observation chunk (D1: never predicted
    chunks) against per-chunk demo features (head-cam region only).

Returned p_hat is INCLUSIVE of the chunk containing the match (i.e. the
number of demo chunks to inject, demo[:p_hat]), so episode start yields
p_hat = 1 (demo chunk 0 visible), not an empty context.
"""
import bisect

import numpy as np
import torch

__all__ = ['GTProgressMatcher', 'LatentRetriever', 'pool_chunk_latent']


class GTProgressMatcher:
    """Ground-truth progress from privileged sim state (client side)."""

    def __init__(self, demo_states, chunk_end_steps, search_ahead_steps=64,
                 drop_time_dim=True):
        demo_states = np.asarray(demo_states, dtype=np.float32)
        # Flattened mujoco states are [time, qpos, qvel]; sim time (col 0)
        # grows monotonically with variance ~2 orders above any pose dim and
        # would reduce "state matching" to timestamp matching — drop it.
        self.drop_time_dim = drop_time_dim
        if drop_time_dim:
            demo_states = demo_states[:, 1:]
        self.demo_states = demo_states
        self.chunk_end_steps = list(chunk_end_steps)
        self.n_chunks = len(self.chunk_end_steps)
        self.search_ahead = search_ahead_steps
        self.prev_step = 0

    def reset(self):
        self.prev_step = 0

    def update(self, sim_state):
        """sim_state: 1-D array (env.get_sim_state()). Returns (p_hat, step)."""
        s = np.asarray(sim_state, dtype=np.float32)
        if self.drop_time_dim:
            s = s[1:]
        lo = self.prev_step
        hi = min(len(self.demo_states), lo + self.search_ahead + 1)
        cand = self.demo_states[lo:hi]
        d = min(cand.shape[1], s.shape[0])  # defensive: dims should match
        dist = np.linalg.norm(cand[:, :d] - s[None, :d], axis=1)
        step = lo + int(dist.argmin())
        self.prev_step = step  # monotonic lower bound for next call
        p_hat = bisect.bisect_left(self.chunk_end_steps, step) + 1
        return min(p_hat, self.n_chunks), step


def pool_chunk_latent(latent, head_cam_width=None):
    """Pool a [1, C, F, H, W] latent chunk into one feature vector.

    Uses only the head-cam (agentview) region: cameras are concatenated along
    W by the server, agentview first, so head-cam = W[:head_cam_width].
    Pooling: mean over F and (H, W) -> [C]-dim feature, per plan section 2.1.
    """
    z = latent.float()
    if head_cam_width is not None:
        z = z[..., :head_cam_width]
    return z.mean(dim=(2, 3, 4)).squeeze(0)  # [C]


class LatentRetriever:
    """Monotonic bounded cosine retrieval over pooled VAE latents (E4)."""

    def __init__(self, demo_latents, chunk_frames=4, head_cam_width=8,
                 search_ahead_chunks=3):
        # demo_latents: [1, C, F_total, H, W]; split into per-chunk features
        n_chunks = demo_latents.shape[2] // chunk_frames
        feats = []
        for i in range(n_chunks):
            chunk = demo_latents[:, :, i * chunk_frames:(i + 1) * chunk_frames]
            feats.append(pool_chunk_latent(chunk, head_cam_width))
        self.demo_feats = torch.nn.functional.normalize(
            torch.stack(feats), dim=-1)  # [n_chunks, C]
        self.n_chunks = n_chunks
        self.head_cam_width = head_cam_width
        self.search_ahead = search_ahead_chunks
        self.prev = 0

    def reset(self):
        self.prev = 0

    def update(self, obs_latent_chunk):
        """obs_latent_chunk: [1, C, F, H, W] latest REAL observation latent."""
        e = torch.nn.functional.normalize(
            pool_chunk_latent(obs_latent_chunk, self.head_cam_width), dim=-1)
        lo = self.prev
        hi = min(self.n_chunks, lo + self.search_ahead + 1)
        sims = self.demo_feats[lo:hi].to(e.device) @ e
        idx = lo + int(sims.argmax().item())
        self.prev = idx
        return idx + 1  # inclusive p_hat
