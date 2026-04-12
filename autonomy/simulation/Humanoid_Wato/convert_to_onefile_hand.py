#!/usr/bin/env python3
"""
convert_to_onefile_hand.py
==========================
Merge all per-episode robot_demos_N.hdf5 files (robomimic-compatible,
written by wato_hand_isaaclab_teleop.py) into a single
demonstrations/robomimic_dataset.hdf5 ready for convert_robomimic_hand.py.

Usage (from the Humanoid_Wato directory):
    python3 convert_to_onefile_hand.py

Or with explicit paths:
    python3 convert_to_onefile_hand.py \
        --input  demonstrations \
        --output demonstrations/robomimic_dataset.hdf5
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="demonstrations",
                   help="Directory containing robot_demos_N.hdf5 files")
    p.add_argument("--output", default="demonstrations/robomimic_dataset.hdf5",
                   help="Output merged HDF5 path")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input)
    output_path = Path(args.output)

    # Collect and sort per-episode HDF5 files
    hdf5_files = sorted(input_dir.glob("robot_demos_*.hdf5"),
                        key=lambda p: int(p.stem.split("_")[-1]))

    if not hdf5_files:
        print(f"[ERROR] No robot_demos_*.hdf5 files found in {input_dir}")
        return

    print(f"[INFO] Found {len(hdf5_files)} episode file(s). Merging → {output_path}")

    env_args = json.dumps({
        "env_name": "IsaacLab-WatoHand",
        "type": 1,
        "env_kwargs": {},
    })

    total_samples = 0
    merged_demos  = 0

    with h5py.File(output_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["env_args"] = env_args

        for i, hdf5_path in enumerate(hdf5_files):
            demo_name = f"demo_{i}"
            try:
                with h5py.File(hdf5_path, "r") as in_f:
                    # New format: data/demo_N/...
                    if "data" in in_f:
                        src_data = in_f["data"]
                        # Find the first demo group inside
                        demo_keys = list(src_data.keys())
                        if not demo_keys:
                            print(f"  [WARN] {hdf5_path.name}: no demo groups found, skipping")
                            continue
                        src_demo = src_data[demo_keys[0]]

                        actions      = np.array(src_demo["actions"])
                        rewards      = np.array(src_demo["rewards"])  if "rewards"  in src_demo else np.zeros(actions.shape[0])
                        dones        = np.array(src_demo["dones"])    if "dones"    in src_demo else np.zeros(actions.shape[0])
                        state        = np.array(src_demo["obs/state"])          if "obs/state"           in src_demo else None
                        ee_poses     = np.array(src_demo["obs/ee_poses"])       if "obs/ee_poses"        in src_demo else None
                        joint_pos    = np.array(src_demo["obs/joint_positions"])if "obs/joint_positions" in src_demo else None

                        # Copy video_path attributes
                        video_attrs = {k: v for k, v in src_demo.attrs.items()
                                       if k.startswith("video_path_")}
                    else:
                        print(f"  [WARN] {hdf5_path.name}: unrecognised schema, skipping")
                        continue

                num_samples = actions.shape[0]
                total_samples += num_samples

                ep_grp = data_grp.create_group(demo_name)
                ep_grp.attrs["num_samples"] = num_samples
                ep_grp.create_dataset("actions",  data=actions.astype(np.float32))
                ep_grp.create_dataset("rewards",  data=rewards.astype(np.float32))
                ep_grp.create_dataset("dones",    data=dones.astype(np.float32))

                obs_grp = ep_grp.create_group("obs")
                if state is not None:
                    obs_grp.create_dataset("state",           data=state.astype(np.float32))
                if ee_poses is not None:
                    obs_grp.create_dataset("ee_poses",        data=ee_poses.astype(np.float32))
                if joint_pos is not None:
                    obs_grp.create_dataset("joint_positions", data=joint_pos.astype(np.float32))

                for attr_key, attr_val in video_attrs.items():
                    ep_grp.attrs[attr_key] = attr_val

                print(f"  {hdf5_path.name} → {demo_name}  ({num_samples} steps, action_dim={actions.shape[1]})")
                merged_demos += 1

            except Exception as e:
                print(f"  [ERROR] Failed to read {hdf5_path.name}: {e}")
                import traceback; traceback.print_exc()

        data_grp.attrs["total"] = total_samples

    print(f"\n[DONE] Merged {merged_demos} demos, {total_samples} total steps → {output_path}")
    print(f"       Re-run convert_robomimic_hand.py next to produce the LeRobot dataset.")


if __name__ == "__main__":
    main()
