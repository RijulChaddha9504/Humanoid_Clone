#!/usr/bin/env python3
"""
convert_robomimic_hand.py
=========================
Convert a robomimic-format HDF5 (produced by convert_to_onefile_hand.py) +
MP4 recordings into a LeRobot v2.0 dataset ready for GR00T fine-tuning.

Camera mapping
--------------
  camera_back      → observation.images.camera_back
  camera_diag_left → observation.images.camera_diag_left
  camera_diag_right→ observation.images.camera_diag_right

Usage (from the Humanoid_Wato directory):
    python3 convert_robomimic_hand.py \\
        --input  demonstrations/robomimic_dataset.hdf5 \\
        --videos recordings \\
        --output demonstrations/lerobot_dataset \\
        --task-description "open door with robotic hand" \\
        --fps 10

The output directory will contain:
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/observation.images.camera_back/episode_XXXXXX.mp4
    videos/chunk-000/observation.images.camera_diag_left/episode_XXXXXX.mp4
    videos/chunk-000/observation.images.camera_diag_right/episode_XXXXXX.mp4
    meta/info.json
    meta/modality.json
    meta/stats.json
    meta/episodes.jsonl
    meta/tasks.jsonl
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNKS_SIZE = 1000

# Camera suffix in recording filenames → LeRobot observation key
CAMERA_KEYS = {
    "camera_back":       "observation.images.camera_back",
    "camera_diag_left":  "observation.images.camera_diag_left",
    "camera_diag_right": "observation.images.camera_diag_right",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",            required=True,
                   help="Path to robomimic_dataset.hdf5")
    p.add_argument("--videos",           required=True,
                   help="Directory containing episode_N_camera_*.mp4 recordings")
    p.add_argument("--output",           required=True,
                   help="Output LeRobot dataset directory")
    p.add_argument("--task-description", default="open door with robotic hand")
    p.add_argument("--fps",              type=int, default=10,
                   help="Recording FPS used during teleop (default 10)")
    return p.parse_args()


def compute_stats(arr):
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
        "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
        "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
    }


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def get_video_res(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
            capture_output=True, text=True,
        )
        for s in json.loads(r.stdout)["streams"]:
            if s.get("codec_type") == "video":
                return s["width"], s["height"]
    except Exception:
        pass
    return 640, 480


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    src      = Path(args.input)
    vid_src  = Path(args.videos)
    dst      = Path(args.output)

    assert src.exists(), f"HDF5 not found: {src}"
    assert vid_src.exists(), f"Videos directory not found: {vid_src}"

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    episodes_meta = []
    all_actions, all_states, all_ee, all_joints = [], [], [], []
    total_frames  = 0
    action_dim    = None
    state_dim     = None
    ee_dim        = 7
    joint_dim     = None

    with h5py.File(src, "r") as f:
        demo_keys = sorted(f["data"].keys())
        n_episodes = len(demo_keys)
        print(f"[INFO] {n_episodes} episode(s) found in {src.name}")

        for ep_idx, key in enumerate(demo_keys):
            g = f["data"][key]

            actions     = g["actions"][:]                # (T, A)
            ee_poses    = g["obs/ee_poses"][:]           # (T, 7)
            joint_pos   = g["obs/joint_positions"][:]    # (T, J)
            state       = g["obs/state"][:]              # (T, S)
            rewards     = g["rewards"][:]
            dones       = g["dones"][:]

            T = actions.shape[0]
            action_dim = actions.shape[1]
            state_dim  = state.shape[1]
            joint_dim  = joint_pos.shape[1]

            print(f"  ep {ep_idx:03d} ({key}): {T} frames | "
                  f"action_dim={action_dim} state_dim={state_dim}")

            chunk_idx = ep_idx // CHUNKS_SIZE
            chunk_dir = dst / f"data/chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            for t in range(T):
                rows.append({
                    "episode_index":  ep_idx,
                    "frame_index":    t,
                    "timestamp":      round(t / args.fps, 6),
                    "task_index":     0,
                    "action":         actions[t].tolist(),
                    "observation.state":           state[t].tolist(),
                    "observation.ee_pose":         ee_poses[t].tolist(),
                    "observation.joint_positions": joint_pos[t].tolist(),
                    "annotation.human.task_description": 0,
                    "next.done":   bool(dones[t]),
                    "next.reward": float(rewards[t]),
                })

            pd.DataFrame(rows).to_parquet(
                chunk_dir / f"episode_{ep_idx:06d}.parquet", index=False
            )

            # Copy videos
            for cam_suffix, obs_key in CAMERA_KEYS.items():
                sv = vid_src / f"episode_{ep_idx}_{cam_suffix}.mp4"
                if not sv.exists():
                    # Also check with original episode numbering (may differ if
                    # episode counter was not reset)
                    print(f"  [WARN] Missing video: {sv.name}")
                    continue
                vd = dst / f"videos/chunk-{chunk_idx:03d}/{obs_key}"
                vd.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sv, vd / f"episode_{ep_idx:06d}.mp4")
                print(f"    → {obs_key}/episode_{ep_idx:06d}.mp4")

            episodes_meta.append({
                "episode_index": ep_idx,
                "tasks":  [args.task_description],
                "length": T,
            })

            all_actions.append(actions)
            all_states.append(state)
            all_ee.append(ee_poses)
            all_joints.append(joint_pos)
            total_frames += T

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "action":                         compute_stats(np.concatenate(all_actions)),
        "observation.state":              compute_stats(np.concatenate(all_states)),
        "observation.ee_pose":            compute_stats(np.concatenate(all_ee)),
        "observation.joint_positions":    compute_stats(np.concatenate(all_joints)),
    }
    with open(dst / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── JSONL meta ────────────────────────────────────────────────────────────
    write_jsonl(dst / "meta/episodes.jsonl", episodes_meta)
    write_jsonl(dst / "meta/tasks.jsonl",
                [{"task_index": 0, "task": args.task_description}])

    # ── modality.json ─────────────────────────────────────────────────────────
    modality = {
        "video": {
            "camera_back": {
                "original_key": "observation.images.camera_back",
            },
            "camera_diag_left": {
                "original_key": "observation.images.camera_diag_left",
            },
            "camera_diag_right": {
                "original_key": "observation.images.camera_diag_right",
            },
        },
        "state": {
            "joint_positions": {
                "start": 0, "end": joint_dim,
                "original_key": "observation.joint_positions",
            },
            "ee_poses": {
                "start": 0, "end": ee_dim,
                "original_key": "observation.ee_pose",
            },
        },
        "action": {
            "arm_wrist_finger": {
                "start": 0, "end": action_dim,
                "original_key": "action",
            },
        },
        "annotation": {
            "human.task_description": {
                "original_key": "annotation.human.task_description",
            },
        },
    }
    with open(dst / "meta/modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # ── info.json ─────────────────────────────────────────────────────────────
    # Try to get resolution from first available video
    w, h = 640, 480
    for cam_suffix in CAMERA_KEYS:
        probe_path = vid_src / f"episode_0_{cam_suffix}.mp4"
        if probe_path.exists():
            w, h = get_video_res(probe_path)
            break

    vf = {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps":          float(args.fps),
            "video.codec":        "h264",
            "video.pix_fmt":      "yuv420p",
            "video.is_depth_map": False,
            "has_audio":          False,
        },
    }

    info = {
        "codebase_version": "v2.0",
        "robot_type":       "custom",
        "total_episodes":   n_episodes,
        "total_frames":     total_frames,
        "total_videos":     len(CAMERA_KEYS),
        "total_chunks":     0,
        "chunks_size":      CHUNKS_SIZE,
        "fps":              float(args.fps),
        "splits":           {"train": f"0:{n_episodes}"},
        "data_path":        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.camera_back":       vf,
            "observation.images.camera_diag_left":  vf,
            "observation.images.camera_diag_right": vf,
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
            },
            "observation.ee_pose": {
                "dtype": "float32",
                "shape": [ee_dim],
            },
            "observation.joint_positions": {
                "dtype": "float32",
                "shape": [joint_dim],
            },
            "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
            "task_index":  {"dtype": "int64",    "shape": [1]},
            "next.done":   {"dtype": "bool",     "shape": [1]},
            "next.reward": {"dtype": "float64",  "shape": [1]},
        },
    }
    with open(dst / "meta/info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n[DONE] {n_episodes} episodes, {total_frames} frames → {dst}")
    print(f"       action_dim={action_dim}  state_dim={state_dim}")
    print(f"       Next step: run train_groot_hand.sh")


if __name__ == "__main__":
    main()
