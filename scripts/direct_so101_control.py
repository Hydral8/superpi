import logging
import sys
import time
from collections import deque

import msgpack_numpy
import numpy as np
import torch
import tyro
import websockets.sync.client
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deg_to_rad(deg):
    return deg * (np.pi / 180.0)

def rad_to_deg(rad):
    return rad * (180.0 / np.pi)

def main(
    port: str = "/dev/tty.usbmodem58760431541",
    camera_index: int = 0,
    server_uri: str = "ws://localhost:8000",
    prompt: str = "pour a cup of coffee",
    fps: int = 30,
):
    # 1. Configuration
    robot_config = So101RobotConfig(
        follower_arms={
            "main": FeetechMotorsBusConfig(
                port=port,
                motors={
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            )
        },
        cameras={
            "laptop": OpenCVCameraConfig(
                camera_index=camera_index,
                fps=fps,
                width=640,
                height=480,
            )
        }
    )
    
    # 2. Initialize Robot
    logger.info(f"Connecting to robot on port {port}...")
    robot = ManipulatorRobot(robot_config)
    try:
        robot.connect()
    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}")
        return
    
    # 3. Connect to Server
    packer = msgpack_numpy.Packer()
    action_queue = deque()
    
    logger.info(f"Connecting to OpenPie server at {server_uri}...")
    try:
        ws = websockets.sync.client.connect(server_uri)
        # Receive metadata
        metadata = msgpack_numpy.unpackb(ws.recv())
        logger.info("Connected to server.")
    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")
        robot.disconnect()
        return

    try:
        logger.info(f"Starting control loop with prompt: '{prompt}'. Press Ctrl+C to stop.")
        while True:
            # Check if we need new actions
            if not action_queue:
                # Capture Observation
                obs_dict = robot.capture_observation()
                
                # Extract image (H, W, C uint8)
                img = obs_dict["observation.images.laptop"].numpy()
                
                # Extract state (Degrees) and convert to Radians
                state_deg = obs_dict["observation.state"].numpy()
                state_rad = deg_to_rad(state_deg)
                
                # Prepare payload
                payload = {
                    "base_0_rgb": img,
                    "state": state_rad,
                    "prompt": prompt
                }
                
                # Query Server
                logger.info("Querying server for new action chunk...")
                ws.send(packer.pack(payload))
                response = ws.recv()
                
                if isinstance(response, str):
                    logger.error(f"Server error: {response}")
                    break
                    
                result = msgpack_numpy.unpackb(response)
                actions_rad = result["actions"] # (Horizon, 6)
                
                # Convert actions back to Degrees and queue
                for a in actions_rad:
                    action_queue.append(rad_to_deg(a))
            
            # Execute next action
            if action_queue:
                action_deg = action_queue.popleft()
                # robot.send_action expects a torch.Tensor
                robot.send_action(torch.from_numpy(action_deg).float())
            
            # Regulate loop speed
            time.sleep(1.0 / fps)
            
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Disconnecting...")
        robot.disconnect()

if __name__ == "__main__":
    tyro.cli(main)
