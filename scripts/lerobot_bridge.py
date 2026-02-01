
import logging
import time
import torch
import numpy as np
from collections import deque
from typing import Dict, Optional

# LeRobot Imports
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so_follower.so100_follower import SO100Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# OpenPie Client Imports
import websockets.sync.client
from openpi_client import msgpack_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsocketLeRobotPolicy:
    """A bridge policy that sends LeRobot observations to the OpenPie server."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, prompt: str = "pour a cup of coffee"):
        self._uri = f"ws://{host}:{port}"
        self._prompt = prompt
        self._packer = msgpack_numpy.Packer()
        self._action_queue = deque()
        
        logger.info(f"Connecting to OpenPie server at {self._uri}...")
        try:
            self._ws = websockets.sync.client.connect(self._uri)
            # Receive initial metadata
            self._metadata = msgpack_numpy.unpackb(self._ws.recv())
            logger.info("Connected to server.")
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise

    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Called by LeRobot record_loop at the robot's FPS."""
        
        if not self._action_queue:
            # 1. Prepare Observation for Server
            # Map 'observation.images.front' -> 'base_0_rgb'
            # Note: Batch is dict of Tensors (B, C, H, W) or similar
            
            # Extract images
            # LeRobot usually provides images as (B, C, H, W) float32 in [0, 1]
            front_img = batch.get("observation.images.front")
            if front_img is not None:
                # Convert back to uint8 (H, W, C) for our server's internal resize logic
                img_np = (front_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = np.zeros((480, 640, 3), dtype=np.uint8)

            # Extract State
            # SO-101 probably has 6 states. Server expects 14-dim ALOHA style.
            state_lerobot = batch.get("observation.state")
            if state_lerobot is not None:
                state_np = state_lerobot[0].cpu().numpy()
                state_14 = np.zeros(14)
                # Map 6-DOF to ALOHA Right Arm positions (indices 7-12 approx)
                # For safety, we just pad to 14.
                state_14[:len(state_np)] = state_np
            else:
                state_14 = np.zeros(14)

            obs = {
                "base_0_rgb": img_np,
                "state": state_14,
                "prompt": self._prompt
            }
            
            # 2. Query Server
            logger.info("Querying server for new action chunk...")
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv()
            
            if isinstance(response, str):
                raise RuntimeError(f"Server Error: {response}")
            
            result = msgpack_numpy.unpackb(response)
            actions = result["actions"] # (50, 6)
            
            # 3. Queue Actions
            # Convert to Tensors
            for a in actions:
                self._action_queue.append(torch.from_numpy(a).float())
                
        # Return next action
        return self._action_queue.popleft()

    def reset(self):
        self._action_queue.clear()

def main():
    FPS = 30
    EPISODE_TIME_SEC = 20
    NUM_EPISODES = 3
    TASK_DESCRIPTION = "pour a cup of coffee"
    
    # Use the user's provided port
    # Note: On Linux it might be /dev/ttyACM0 or similar, but we'll try their exact string.
    ROBOT_PORT = "/dev/tty.usbmodem58760434471"
    
    # Create the robot configuration
    camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    robot_config = SO100FollowerConfig(
        port=ROBOT_PORT, id="so101_arm", cameras=camera_config
    )

    # Initialize the robot
    logger.info("Initializing Robot...")
    robot = SO100Follower(robot_config)
    robot.connect()

    # Initialize the bridge policy
    logger.info("Connecting to OpenPie Bridge...")
    policy = WebsocketLeRobotPolicy(host="localhost", port=8000, prompt=TASK_DESCRIPTION)

    # Dataset/Visualization setup (using user's logic)
    HF_DATASET_ID = "local_eval_dataset"
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=HF_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
    )

    _, events = init_keyboard_listener()
    # init_rerun(session_name="recording") # Optional: can be noisy in terminal

    for episode_idx in range(NUM_EPISODES):
        log_say(f"Running inference {episode_idx + 1}/{NUM_EPISODES}")
        
        policy.reset()
        
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        dataset.save_episode()

    robot.disconnect()
    logger.info("Task complete.")

if __name__ == "__main__":
    main()
