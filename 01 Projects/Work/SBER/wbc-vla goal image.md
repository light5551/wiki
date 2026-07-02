
branch: `sg/train_goal_image`
## How to run train?

## Example
```bash
HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface HF_HUB_CACHE=~/.cache/huggingface/hub HF_LEROBOT_HOME=~/.cache/huggingface/lerobot OMP_NUM_THREADS=1 TRITON_CACHE_DIR=/tmp/triton_cache_jovyan TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 CLEARML_CONFIG_FILE=/home/jovyan/kozhukovv/lerobot-fork/clearml.conf accelerate launch --config_file accelerate_config_fsdp2.yaml lerobot/scripts/train_with_validation.py --config-name=humanoid_nav_rosbags_mixed_2b
```

My Launch