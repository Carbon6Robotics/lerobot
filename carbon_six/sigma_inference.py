import inspect
import time
from pathlib import Path

import drake
import hydra
import numpy as np
import pydrake.lcm
import torch
from omegaconf import DictConfig, OmegaConf

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


class LcmDataInterface:
    """Interface for managing subscribed data and publishing commands"""

    def handle_camera_left_wrist(self, data) -> None:
        msg = drake.lcmt_image_array.decode(data)
        self.left_image = np.frombuffer(msg.images[0].data, dtype=np.uint8).reshape(
            (msg.images[0].height, msg.images[0].width, 3)
        )
        # cv2.cvtColor(
        #     np.frombuffer(msg.images[0].data, dtype=np.uint8).reshape(
        #         (msg.images[0].height, msg.images[0].width, 3)
        #     ),
        #     cv2.COLOR_BGR2RGB,
        # )
        print(f"{msg.header.utime}: {inspect.currentframe().f_code.co_name}")
        return

    def handle_camera_right_wrist(self, data) -> None:
        msg = drake.lcmt_image_array.decode(data)
        self.right_image = np.frombuffer(msg.images[0].data, dtype=np.uint8).reshape(
            (msg.images[0].height, msg.images[0].width, 3)
        )
        print(f"{msg.header.utime}: {inspect.currentframe().f_code.co_name}")
        return

    def handle_status_left(self, data):
        msg = drake.lcmt_iiwa_status.decode(data)
        self.state[0:6] = msg.joint_position_measured
        print(f"{msg.utime}: {inspect.currentframe().f_code.co_name}")
        return

    def handle_status_right(self, data):
        msg = drake.lcmt_iiwa_status.decode(data)
        self.state[7:13] = msg.joint_position_measured
        print(f"{msg.utime}: {inspect.currentframe().f_code.co_name}")
        return

    def handle_shunk_status_left(self, data):
        msg = drake.lcmt_schunk_wsg_status.decode(data)
        self.state[6] = msg.actual_position_mm
        print(f"{msg.utime}: {inspect.currentframe().f_code.co_name}")
        return

    def handle_shunk_status_right(self, data):
        msg = drake.lcmt_schunk_wsg_status.decode(data)
        self.state[13] = msg.actual_position_mm
        print(f"{msg.utime}: {inspect.currentframe().f_code.co_name}")
        return

    def __init__(self, timeout: int = 0) -> None:
        self.lcm = pydrake.lcm.DrakeLcm()
        self.timeout_milliseconds = int(timeout)

        self.left_image = None
        self.right_image = None
        self.state = np.zeros((14,))

        # Set up subscribers
        self.lcm.Subscribe("CAMERA_1", self.handle_camera_left_wrist)
        self.lcm.Subscribe("CAMERA_2", self.handle_camera_right_wrist)
        self.lcm.Subscribe("DSR_STATUS_LEFT", self.handle_status_left)
        self.lcm.Subscribe("DSR_STATUS_RIGHT", self.handle_status_right)
        self.lcm.Subscribe("SCHUNK_EGK_STATUS_LEFT", self.handle_shunk_status_left)
        self.lcm.Subscribe("SCHUNK_EGK_STATUS_RIGHT", self.handle_shunk_status_right)

    def handle_subscriptions(self) -> int:
        return self.lcm.HandleSubscriptions(self.timeout_milliseconds)

    def publish(self, action) -> None:
        """Publish action

        Args:
            action (numpy, dim 14): [left joint cmd, left gripper, right joint cmd, right gripper]
        """

        # left arm
        iiwa_command = drake.lcmt_iiwa_command()
        iiwa_command.num_joints = 6
        iiwa_command.num_torques = 6
        iiwa_command.joint_position = action[0:6]
        iiwa_command.joint_torque = np.zeros(6)
        self.lcm.Publish(channel="DSR_COMMAND_LEFT", buffer=iiwa_command.encode())

        # left gripper
        schunk_command = drake.lcmt_schunk_wsg_command()
        schunk_command.target_position_mm = action[6]
        self.lcm.Publish(channel="SCHUNK_EGK_COMMAND_LEFT", buffer=schunk_command.encode())

        # right arm
        iiwa_command = drake.lcmt_iiwa_command()
        iiwa_command.num_joints = 6
        iiwa_command.num_torques = 6
        iiwa_command.joint_position = action[7:13]
        iiwa_command.joint_torque = np.zeros(6)
        self.lcm.Publish(channel="DSR_COMMAND_RIGHT", buffer=iiwa_command.encode())

        # right gripper
        schunk_command = drake.lcmt_schunk_wsg_command()
        schunk_command.target_position_mm = action[13]
        self.lcm.Publish(channel="SCHUNK_EGK_COMMAND_RIGHT", buffer=schunk_command.encode())

        print("\nL: ", end="")
        for a in action[0:7]:
            print(f" {a:+8.4f} ", end="")
        print("\nR: ", end="")
        for a in action[7:14]:
            print(f" {a:+8.4f} ", end="")

    def is_data_valid(self) -> bool:
        if self.state is None:
            return False

        if self.left_image is None:
            return False

        if self.right_image is None:  # noqa: SIM103
            return False

        return True


@hydra.main(version_base="1.3", config_name="sigma_config", config_path="./config/inference")
def eval(cfg: DictConfig):
    """
    Use pre-trained policy to do evaluation
    """

    ####
    # Load pretrained model
    ####
    policy = DiffusionPolicy.from_pretrained(cfg.model)
    policy.eval()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
        policy.diffusion.num_inference_steps = 10

    policy.to(device)
    policy.reset()

    ####
    # Set Up LCM
    ####
    lcm_data = LcmDataInterface((1000 / cfg.fps) * 3)

    ####
    # Do inference
    ####
    while True:
        start = time.perf_counter_ns()
        lcm_data.handle_subscriptions()
        start_pred = time.perf_counter_ns()

        if not lcm_data.is_data_valid():
            continue

        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(lcm_data.state)
        left_image = torch.from_numpy(lcm_data.left_image)
        right_image = torch.from_numpy(lcm_data.right_image)

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)

        left_image = left_image.to(torch.float32) / 255
        left_image = left_image.permute(2, 0, 1)

        right_image = right_image.to(torch.float32) / 255
        right_image = right_image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        left_image = left_image.to(device, non_blocking=True)
        right_image = right_image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.left_wrist_rgb": left_image,
            "observation.images.right_wrist_rgb": right_image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Publish data
        lcm_data.publish(numpy_action)

        end = time.perf_counter_ns()
        print(f"Start -> Predcit : {(start_pred - start)/1e9}")
        print(f"Prediction       : {(end - start_pred)/1e9}")
        print(f"Total Duration[s]: {(end - start)/1e9}")
        time.sleep(1 / cfg.fps)


if __name__ == "__main__":
    eval()
