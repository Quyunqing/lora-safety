import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import argparse
from pathlib import Path

def load_lora_pipeline(base_model_path: str, lora_weights_path: str):
    """加载基础模型并合并LoRA权重"""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,  # 注意：我们是在做安全研究，可能需要禁用默认安全检查器
    ).to("cuda")
    
    # 加载LoRA权重
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights_path)
    
    return pipe

def generate_images(pipe, prompts, output_dir, num_images_per_prompt=4):
    """生成图像"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        for j in range(num_images_per_prompt):
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                seed=42 + j
            ).images[0]
            
            save_path = output_dir / f"prompt_{i:03d}_sample_{j:03d}.png"
            image.save(save_path)
            print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_weights", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--prompts_file", type=str, required=True, help="File containing prompts (one per line)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=4)
    args = parser.parse_args()
    
    # 读取prompts
    with open(args.prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # 加载模型
    pipe = load_lora_pipeline(args.base_model, args.lora_weights)
    
    # 生成图像
    generate_images(pipe, prompts, args.output_dir, args.num_images)
    
    print(f"All images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
