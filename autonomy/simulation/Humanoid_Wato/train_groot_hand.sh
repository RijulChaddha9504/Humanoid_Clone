#!/bin/bash
# ============================================================
# train_groot_hand.sh
# GR00T N1.6 fine-tuning for the WATO robotic hand
# ============================================================
# SLURM config
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_3090:2,tmpdisk:102400
#SBATCH --time=24:00:00
#SBATCH --job-name=groot_hand_training
#SBATCH --output=/home/rijul_chaddha/groot_hand_training_%j.log

# Your HuggingFace token — set this before submitting:
# export HF_TOKEN="hf_XXXX"   ← run this in your shell before sbatch

# ── 1. Start Docker daemon ────────────────────────────────────────────────────
slurm-start-dockerd.sh
export DOCKER_HOST=unix:///tmp/run/docker.sock
sleep 60

# ── 2. Start Isaac Lab container ──────────────────────────────────────────────
cd ~/IsaacLab
./docker/container.py start ros2 &
START_PID=$!

echo "Waiting for container to start..."
for i in $(seq 1 120); do
    if docker ps | grep -q "isaac-lab-ros2"; then
        echo "Container is running!"
        break
    fi
    echo "Waiting... ($i/120)"
    sleep 10
done
wait $START_PID
sleep 60

# ── 3. Inner script (runs inside Docker) ─────────────────────────────────────
cat > /tmp/inner_train_hand.sh << 'INNEREOF'
#!/bin/bash
set -e

REPO_ROOT=/workspace/isaaclab/humanoid/autonomy/simulation/Humanoid_Wato
LEROBOT_DATASET=$REPO_ROOT/demonstrations/lerobot_dataset
CHECKPOINT_DIR=$REPO_ROOT/checkpoints

echo "============================================================"
echo "  WATO Hand — GR00T Imitation Learning Training"
echo "============================================================"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
echo "[1/8] Pulling latest code..."
cd /workspace/isaaclab/humanoid && git pull origin main || true
sleep 30

# ── 2. Install Python dependencies ───────────────────────────────────────────
echo "[2/8] Installing dependencies..."
/isaac-sim/kit/python/bin/python3 -m pip install \
    tyro filelock packaging pydantic mpmath psutil \
    python-dateutil pytz Pillow wheel h5py pandas pyarrow
sleep 30

# ── 3. Install GR00T (editable) ──────────────────────────────────────────────
echo "[3/8] Installing Isaac GR00T..."
cd /workspace/Isaac-GR00T && /isaac-sim/kit/python/bin/python3 -m pip install -e .
sleep 30

# ── 4. Fix nvcc stub ─────────────────────────────────────────────────────────
echo "[4/8] Creating nvcc stub..."
mkdir -p /usr/local/cuda/bin
cat > /usr/local/cuda/bin/nvcc << 'NVCC'
#!/bin/bash
echo nvcc: NVIDIA R Cuda compiler driver
echo Cuda compilation tools, release 12.2, V12.2.91
NVCC
chmod +x /usr/local/cuda/bin/nvcc
sleep 10

# ── 5. Clear stale HuggingFace cache ─────────────────────────────────────────
echo "[5/8] Clearing stale HuggingFace cache..."
rm -rf /root/.cache/huggingface/modules
sleep 10

# ── 6. Set env variables ──────────────────────────────────────────────────────
echo "[6/8] Setting environment variables..."
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1

# ── 7. Convert demonstrations (if lerobot_dataset doesn't exist yet) ─────────
echo "[7/8] Checking dataset..."
if [ ! -d "$LEROBOT_DATASET/meta" ]; then
    echo "  LeRobot dataset not found — running conversion scripts..."
    cd $REPO_ROOT

    echo "  Step A: Merging per-episode HDF5 files..."
    /isaac-sim/kit/python/bin/python3 convert_to_onefile_hand.py \
        --input  demonstrations \
        --output demonstrations/robomimic_dataset.hdf5

    echo "  Step B: Converting to LeRobot v2.0 format..."
    /isaac-sim/kit/python/bin/python3 convert_robomimic_hand.py \
        --input  demonstrations/robomimic_dataset.hdf5 \
        --videos recordings \
        --output demonstrations/lerobot_dataset \
        --task-description "open door with robotic hand" \
        --fps 10
else
    echo "  LeRobot dataset found at $LEROBOT_DATASET — skipping conversion."
fi

# ── 8. Train ──────────────────────────────────────────────────────────────────
echo "[8/8] Starting GR00T training..."
mkdir -p $CHECKPOINT_DIR
cd /workspace/Isaac-GR00T

TRANSFORMERS_ATTN_IMPLEMENTATION=eager \
/isaac-sim/kit/python/bin/python3 -m torch.distributed.run \
    --nproc_per_node=2 \
    gr00t/experiment/launch_finetune.py \
        --base-model-path        nvidia/GR00T-N1.6-3B \
        --dataset-path           $LEROBOT_DATASET \
        --embodiment-tag         NEW_EMBODIMENT \
        --modality-config-path   /workspace/Isaac-GR00T/gr00t/configs/data/custom_embodiment.py \
        --output-dir             $CHECKPOINT_DIR \
        --global-batch-size      8 \
        --gradient-accumulation-steps 2 \
        --learning-rate          1e-4 \
        --max-steps              5000 \
        --dataloader-num-workers 4 \
        --no-tune-llm \
        --no-tune-visual \
        --tune-projector \
        --tune-diffusion-model \
        --num-gpus               2 \
        --save-steps             5000

# ── 9. Upload to HuggingFace ──────────────────────────────────────────────────
echo "Training complete! Uploading to HuggingFace..."
/isaac-sim/kit/python/bin/python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(
    folder_path='$CHECKPOINT_DIR',
    repo_id='Ultrox9504/wato_hand_gr00t_weights',
    repo_type='model',
    ignore_patterns=['**/global_step*', '**/zero_*', '**/mp_rank*'],
    commit_message='WATO hand training -- 5000 steps',
)
print('Upload complete!')
"
INNEREOF

# ── 4. Copy and execute inner script in Docker ────────────────────────────────
docker cp /tmp/inner_train_hand.sh isaac-lab-ros2:/tmp/inner_train_hand.sh
docker exec isaac-lab-ros2 bash /tmp/inner_train_hand.sh
