generate_configs.py
from config_system import LoRAConfig
import os

# 定义8组不同的层选择策略
EXPERIMENTS = {
    "exp01_only_q": {
        "experiment_name": "only_q",
        "target_modules": ["to_q"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp01_only_q",
    },
    "exp02_only_k": {
        "experiment_name": "only_k", 
        "target_modules": ["to_k"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp02_only_k",
    },
    "exp03_only_v": {
        "experiment_name": "only_v",
        "target_modules": ["to_v"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp03_only_v",
    },
    "exp04_qkv": {
        "experiment_name": "qkv",
        "target_modules": ["to_q", "to_k", "to_v"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp04_qkv",
    },
    "exp05_resnet": {
        "experiment_name": "resnet",
        "target_modules": ["conv1", "conv2", "conv_shortcut", "time_emb_proj"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp05_resnet",
    },
    "exp06_skip": {
        "experiment_name": "skip",
        "target_modules": ["conv_shortcut"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp06_skip",
    },
    "exp07_full_attn": {
        "experiment_name": "full_attn",
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp07_full_attn",
    },
    "exp08_all": {
        "experiment_name": "all_layers",
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "conv1", "conv2", "conv_shortcut", "time_emb_proj", "proj_in", "proj_out"],
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 200,
        "output_dir": "outputs/exp08_all",
    },
}

def generate_all_configs():
    os.makedirs("configs", exist_ok=True)
    for exp_id, config_dict in EXPERIMENTS.items():
        config = LoRAConfig(**config_dict)
        config.to_yaml(f"configs/{exp_id}.yaml")
        print(f"Generated: configs/{exp_id}.yaml")

if __name__ == "__main__":
    generate_all_configs()
