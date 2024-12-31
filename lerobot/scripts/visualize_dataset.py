#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import logging

# import time
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.utils.data
import tqdm

# import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualization_helper.rerun_loader_urdf import URDFLogger
from lerobot.scripts.visualization_helper.urdf_helper import link_to_world_transform, log_angle_rot


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def construct_blueprints(dataset: LeRobotDataset, show_images: bool) -> rrb.Blueprint:
    tabs = []

    # Tab for images
    if show_images:
        tabs = [
            rrb.Grid(
                *[
                    rrb.Spatial2DView(
                        origin=key.replace(".", "/"),  # join upto last part
                        contents=key.replace(".", "/"),  # full entity path
                    )
                    for key in dataset.meta.camera_keys
                ],
                name="Camera Images",
            )
        ]

    # Tab for video
    tabs.append(
        rrb.Grid(
            *[
                rrb.Spatial2DView(
                    origin=key.replace(".", "/").replace("images", "videos"),
                    contents=key.replace(".", "/").replace("images", "videos"),
                )
                for key in dataset.meta.video_keys
            ],
            name="Camera Videos",
        )
    )

    # Action
    tabs.append(
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="/action",
                contents="+ /action/SCHUNK_EGK_COMMAND_LEFT::position_00\n+ /state/SCHUNK_EGK_STATUS_LEFT::position_00",
                name="Gripper Left",
            ),
            rrb.TimeSeriesView(
                origin="/action",
                contents="+ /action/SCHUNK_EGK_COMMAND_RIGHT::position_00\n+ /state/SCHUNK_EGK_STATUS_RIGHT::position_00",
                name="Gripper Right",
            ),
            # grid_columns=1,
            name="Gripper",
        )
    )

    # Next
    tabs.append(rrb.TimeSeriesView(origin="/next", contents="/next/**", name="Next"))

    # State
    tabs.append(
        rrb.Grid(
            *[
                rrb.TimeSeriesView(
                    origin="/state",
                    contents=f"+ /state/DSR_STATUS_LEFT::position_0{i}\n+ /action/DSR_COMMAND_LEFT::position_0{i}",
                    name=f"LEFT Joint {i} State & Action",
                )
                for i in range(0, 6)
            ],
            *[
                rrb.TimeSeriesView(
                    origin="/state",
                    contents=f"+ /state/DSR_STATUS_RIGHT::position_0{i}\n+ /action/DSR_COMMAND_RIGHT::position_0{i}",
                    name=f"RIGHT Joint {i} State & Action",
                )
                for i in range(0, 6)
            ],
            name="Joint",
        )
    )

    # 3D-model
    tabs.append(rrb.Spatial3DView(contents="/**", name="Robot"))

    # Construct blueprint
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            *tabs,
            name="Data",
            active_tab=1 if show_images else 0,  # always default to video tab
        ),
    )

    return blueprint


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    model_dir: Path | None = None,
    show_images: bool = False,
    # save: bool = False,
    # output_dir: Path | None = None,
) -> Path | None:
    # if save:
    #     assert (
    #         output_dir is not None
    #     ), "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    # Setting up rerun blueprints
    my_blueprint = construct_blueprints(dataset, show_images)

    rr.init(f"{repo_id}/episode_{episode_index:06}", default_blueprint=my_blueprint)
    rr.serve_web(
        open_browser=True,
        web_port=web_port,
        ws_port=ws_port,
        default_blueprint=my_blueprint,
    )

    logging.info("Logging to Rerun")

    urdf_logger = URDFLogger(str((Path(model_dir) / "m0609.urdf").resolve()))
    urdf_logger.log()
    # print(urdf_logger.entity_to_transform)

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # display each camera image
            if show_images:
                for key in dataset.meta.camera_keys:
                    rr.log(key.replace(".", "/"), rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    motor_name = dataset.meta.features["action"]["names"]["motors"][dim_idx]
                    rr.log(f"action/{motor_name}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    motor_name = dataset.meta.features["observation.state"]["names"]["motors"][dim_idx]
                    rr.log(f"state/{motor_name}", rr.Scalar(val.item()))

                joint_angles = (batch["observation.state"][i][0:6]).tolist()
                # print("--------")
                # print(joint_angles)
                joint_origins = []
                for joint_idx, angle in enumerate(joint_angles):
                    # Log robot's model
                    transform = link_to_world_transform(
                        urdf_logger.entity_to_transform, joint_angles, joint_idx + 1
                    )
                    joint_org = (transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
                    joint_origins.append(joint_org)

                    log_angle_rot(urdf_logger.entity_to_transform, joint_idx + 1, angle)

            if "next.done" in batch:
                rr.log("next/done", rr.Scalar(batch["next.done"][i].item()))

            if "next.reward" in batch:
                rr.log("next/reward", rr.Scalar(batch["next.reward"][i].item()))

            if "next.success" in batch:
                rr.log("next/success", rr.Scalar(batch["next.success"][i].item()))

    # Log video asset which is referred to by frame references.
    for video_key in dataset.meta.video_keys:
        video_asset = rr.AssetVideo(
            path=str(dataset.root)
            + "/"
            + dataset.meta.video_path.format(
                **{"episode_chunk": 0, "video_key": video_key, "episode_index": episode_index}
            )
        )
        rr.log(f"{video_key.replace('.','/').replace('images','videos')}", video_asset, static=True)

        # Send automatically determined video frame timestamps.
        frame_timestamps_ns = video_asset.read_frame_timestamps_ns()
        rr.send_columns(
            f"{video_key.replace('.','/').replace('images','videos')}",
            # Note timeline values don't have to be the same as the video timestamps.
            times=[rr.TimeNanosColumn("timestamp", frame_timestamps_ns)],
            components=[
                rr.VideoFrameReference.indicator(),
                rr.components.VideoTimestamp.nanoseconds(frame_timestamps_ns),
            ],
        )

    rr.send_blueprint(blueprint=my_blueprint)

    # if mode == "local":
    #     # save .rrd locally
    #     output_dir = Path(output_dir)
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     repo_id_str = repo_id.replace("/", "_")
    #     rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
    #     rr.save(rrd_path)
    #     return rrd_path
    # elif mode == "distant":
    #     # stop the process from exiting since it is serving the websocket connection
    #     try:
    #         while True:
    #             time.sleep(1)
    #     except KeyboardInterrupt:
    #         print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--local-files-only",
        type=int,
        default=1,
        help="Use local files only. By default, this script will try to use the local data.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    # parser.add_argument(
    #     "--output-dir",
    #     type=Path,
    #     default=None,
    #     help="Directory path to write a .rrd file when `--save 1` is set.",
    # )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="distant",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to model directory, e.g., ./models",
    )
    parser.add_argument(
        "--show-images",
        # type=bool,
        default=False,
        action="store_true",
        help="Rerun to show individual images from the dataset. This takes a lot of memory and time to load!",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Allowed tolerance on the discrepency between desired timestamp and retrieved image from the video",
    )

    # parser.add_argument(
    #     "--save",
    #     type=int,
    #     default=0,
    #     help=(
    #         "Save a .rrd file in the directory provided by `--output-dir`. "
    #         "It also deactivates the spawning of a viewer. "
    #         "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
    #     ),
    # )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance = kwargs.pop("tolerance")
    local_files_only = kwargs.pop("local_files_only")

    # Load data set
    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, root=root, local_files_only=local_files_only, tolerance_s=tolerance)

    # Visualize data
    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
