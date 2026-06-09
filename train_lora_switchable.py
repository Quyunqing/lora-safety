import os
import argparse
import torch
from pathlib import Path
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from config_system import LoRAConfig
import wandb

class LoRATrainer:
    def __init__(self, config_path: str):
        self.config = LoRAConfig.from_yaml(config_path)
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=1,
        )
        
        # 初始化wandb记录
        if self.accelerator.is_main_process:
            wandb.init(
                project="lora-safety-alignment",
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
        
        self.setup_model()
        self.setup_optimizer()
        
    def setup_model(self):
        """加载模型并插入LoRA"""
        # 加载基础组件
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.base_model, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model, subfolder="unet"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.base_model, subfolder="scheduler"
        )
        
        # 关键：根据配置选择target_modules插入LoRA
        lora_config = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,  # 从配置读取！
            lora_dropout=0.0,
            bias="none",
        )
        
        # 只在UNet上应用LoRA
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()  # 打印可训练参数量
        
        # 冻结其他组件
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # 准备accelerator
        self.unet, self.text_encoder, self.vae = self.accelerator.prepare(
            self.unet, self.text_encoder, self.vae
        )
        
    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)
        
    def train_step(self, batch):
        """单步训练"""
        # 这里需要根据你的具体任务实现训练逻辑
        # 如果是概念抑制任务，需要加载概念数据集
        pass
        
    def save_checkpoint(self, step: int):
        """保存LoRA权重"""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存LoRA权重
        self.unet.save_pretrained(output_dir / "lora_weights")
        
        # 同时保存配置文件
        self.config.to_yaml(output_dir / "config.yaml")
        
        if self.accelerator.is_main_process:
            print(f"Saved checkpoint to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    trainer = LoRATrainer(args.config)
    
    # 训练循环
    for step in range(trainer.config.num_train_steps):
        # ... 训练逻辑
        if step % trainer.config.save_steps == 0:
            trainer.save_checkpoint(step)
    
    # 保存最终模型
    trainer.save_checkpoint("final")

if __name__ == "__main__":
    main()
