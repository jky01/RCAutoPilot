"""Observation wrapper to extract lane lines using CLRNet."""

from __future__ import annotations

import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import cv2
    import torch
    from clrnet.utils.config import Config
    from clrnet.models.registry import build_net
    from clrnet.utils.net_utils import load_network
except Exception:  # pragma: no cover - CLRNet is optional
    cv2 = None
    torch = None


class CLRLaneDetectionWrapper(gym.ObservationWrapper):
    """Wrap environment to replace images with lane masks."""

    def __init__(self, env: gym.Env, config_path: str | None = None, checkpoint_path: str | None = None, device: str = "cpu"):
        super().__init__(env)
        self.device = device
        self.model = None
        if config_path is None:
            config_path = os.environ.get("LANE_CFG")
        if checkpoint_path is None:
            checkpoint_path = os.environ.get("LANE_CKPT")

        if config_path and checkpoint_path and cv2 is not None and torch is not None:
            cfg = Config.fromfile(config_path)
            self.resize = (cfg.img_w, cfg.img_h)
            self.cut_height = getattr(cfg, "cut_height", 0)
            self.img_norm = cfg.img_norm
            self.model = build_net(cfg)
            load_network(self.model, checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            self.cfg = cfg
        else:
            obs_shape = env.observation_space.shape
            self.resize = (obs_shape[1], obs_shape[0])
            self.cut_height = 0
            self.img_norm = {"mean": [0, 0, 0], "std": [1, 1, 1]}
            self.cfg = None

        # observation is single channel mask
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.resize[1], self.resize[0], 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.model is None or cv2 is None or torch is None:
            return observation

        img = cv2.resize(observation, self.resize)
        if self.cut_height > 0:
            img = img[self.cut_height :, :, :]
        img = img.astype(np.float32)
        mean = np.array(self.img_norm["mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.img_norm["std"], dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model({"img": tensor})
            lanes = self.model.heads.get_lanes(output)[0]

        mask = np.zeros((self.resize[1], self.resize[0]), dtype=np.uint8)
        for lane in lanes:
            pts = lane.to_array(self.cfg).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=2)

        mask = mask[:, :, None]
        if self.cut_height > 0:
            pad = np.zeros((self.cut_height, self.resize[0], 1), dtype=np.uint8)
            mask = np.vstack((pad, mask))
        return mask
