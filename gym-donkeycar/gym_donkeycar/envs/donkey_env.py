"""
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
from gym_donkeycar.lane_detector import LaneDetector

logger = logging.getLogger(__name__)


def supply_defaults(conf: Dict[str, Any]) -> None:
    """
    Update the config dictonnary
    with defaults when values are missing.

    :param conf: The user defined config dict,
        passed to the environment constructor.
    """
    defaults = [
        ("start_delay", 5.0),
        ("max_cte", 8.0),
        ("frame_skip", 1),
        ("cam_resolution", (120, 160, 3)),
        ("log_level", logging.INFO),
        ("host", "localhost"),
        ("port", 9091),
        ("steer_limit", 1.0),
        ("throttle_min", 0.0),
        ("throttle_max", 1.0),
        # Enable screenshot capture by default every two seconds into
        # the ``images`` directory so users can inspect raw frames.
        #("screenshot_interval", 2),
        #("screenshot_dir", "images"),
    ]

    for key, val in defaults:
        if key not in conf:
            conf[key] = val
            print(f"Setting default: {key} {val}")


class DonkeyEnv(gym.Env):
    """OpenAI Gym Environment for Donkey.

    :param level: name of the level to load
    :param conf: configuration dictionary
    :param render_mode: rendering mode passed by gymnasium (ignored)
    :param lane_detection: enable simple lane detection preprocessing
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    ACTION_NAMES: List[str] = ["steer", "throttle"]
    VAL_PER_PIXEL: int = 255

    def __init__(
        self,
        level: str,
        conf: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        lane_detection: Optional[bool] = None,
    ) -> None:
        print("starting DonkeyGym env")
        self.viewer = None
        self.proc = None
        self.render_mode = render_mode

        if conf is None:
            conf = {}

        conf["level"] = level
        if lane_detection is not None:
            conf["lane_detection"] = lane_detection

        # ensure defaults are supplied if missing.
        supply_defaults(conf)

        # set logging level
        logging.basicConfig(level=conf["log_level"])

        logger.debug("DEBUG ON")
        logger.debug(conf)

        # start Unity simulation subprocess
        self.proc = None
        if "exe_path" in conf:
            self.proc = DonkeyUnityProcess()
            # the unity sim server will bind to the host ip given
            self.proc.start(conf["exe_path"], host="0.0.0.0", port=conf["port"])

            # wait for simulator to startup and begin listening
            time.sleep(conf["start_delay"])

        # start simulation com
        lane_detector = LaneDetector() if conf.get("lane_detection") else None
        self.viewer = DonkeyUnitySimContoller(conf=conf, lane_detector=lane_detector)

        # Note: for some RL algorithms, it would be better to normalize the action space to [-1, 1]
        # and then rescale to proper limtis
        # steering and throttle
        self.action_space = spaces.Box(
            low=np.array([-float(conf["steer_limit"]), float(conf["throttle_min"])]),
            high=np.array([float(conf["steer_limit"]), float(conf["throttle_max"])]),
            dtype=np.float32,
        )

        # camera sensor data
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = conf["frame_skip"]

        # wait until the car is loaded in the scene
        self.viewer.wait_until_loaded()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.quit()
        if hasattr(self, "proc") and self.proc is not None:
            self.proc.quit()

    def set_reward_fn(self, reward_fn: Callable) -> None:
        self.viewer.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn: Callable) -> None:
        self.viewer.set_episode_over_fn(ep_over_fn)

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an action in the environment using the new Gym API."""
        for _ in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.viewer.observe()

        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment using the new Gym API."""
        if seed is not None:
            self.seed(seed)
        # Activate hand brake, so the car does not move
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.1)
        self.viewer.reset()
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.1)
        observation, _, _, info = self.viewer.observe()
        return observation, info

    def render(self, mode: str = "human", close: bool = False) -> Optional[np.ndarray]:
        """Render the environment using the internal viewer."""
        if close:
            self.viewer.quit()

        # When called through a VecEnv, ``mode`` can be ``None``.
        # Fallback to the env ``render_mode`` when provided to ensure we return
        # an RGB array for video recording.
        if mode is None:
            mode = self.render_mode or "human"

        # Gymnasium expects the mode argument to match the environment
        # ``render_mode``. If the caller passes "human" while the environment was
        # created with ``render_mode='rgb_array'``, we override it so that the
        # frame is returned instead of nothing.
        if mode == "human" and self.render_mode == "rgb_array":
            mode = "rgb_array"

        return self.viewer.render(mode)

    def is_game_over(self) -> bool:
        return self.viewer.is_game_over()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class GeneratedRoadsEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="generated_road", *args, **kwargs)


class WarehouseEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="warehouse", *args, **kwargs)


class AvcSparkfunEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="sparkfun_avc", *args, **kwargs)


class GeneratedTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="generated_track", *args, **kwargs)


class MountainTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="mountain_track", *args, **kwargs)


class RoboRacingLeagueTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="roboracingleague_1", *args, **kwargs)


class WaveshareEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="waveshare", *args, **kwargs)


class MiniMonacoEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="mini_monaco", *args, **kwargs)


class WarrenTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="warren", *args, **kwargs)


class ThunderhillTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="thunderhill", *args, **kwargs)


class CircuitLaunchEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="circuit_launch", *args, **kwargs)
