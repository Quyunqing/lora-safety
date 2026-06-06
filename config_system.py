from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class LoRAConfig:
    experiment_name: str
    target_modules: List[str]  # 关键：层选择开关
    rank: int
    lora_alpha: int
    learning_rate: float
    num_train_steps: int
    save_steps: int
    output_dir: str
    base_model: str = "runwayml/stable-diffusion-v1-5"
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
