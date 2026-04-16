from huggingface_hub import HfApi
import os

# --- EDIT THESE TWO VARIABLES BEFORE RUNNING ---
HF_TOKEN = os.getenv("HF_TOKEN")  # Replace with your HuggingFace Write Token
REPO_ID = "Ultrox9504/wato_hand_demonstrations"  # Replace if your username or repo name is different
# -----------------------------------------------

FOLDER_PATH = r"C:\Users\rijul\Downloads\UWaterloo\WATonomous\humanoid\autonomy\simulation\Humanoid_Wato\demonstrations\lerobot_dataset"

print(f"Uploading {FOLDER_PATH} to https://huggingface.co/datasets/{REPO_ID}...")

api = HfApi(token=HF_TOKEN)
try:
    api.upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Initial WATO hand dataset - 10 episodes",
    )
    print("\n✅ Dataset uploaded successfully!")
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("Double check that your token is correct and has 'Write' permissions, and that the repository exists.")
