# 对所有训练好的模型进行批量推理
OUTPUT_BASE="outputs"
INFERENCE_OUTPUT="inference_results"
PROMPTS_FILE="test_prompts.txt"  # 由成员C提供

mkdir -p $INFERENCE_OUTPUT

# 为每个实验生成图像
for exp_dir in $OUTPUT_BASE/*/; do
    exp_name=$(basename $exp_dir)
    lora_weights="$exp_dir/checkpoint-final/lora_weights"
    
    if [ ! -d "$lora_weights" ]; then
        echo "Skipping $exp_name: LoRA weights not found"
        continue
    fi
    
    echo "Running inference for $exp_name..."
    python inference_with_lora.py \
        --lora_weights $lora_weights \
        --prompts_file $PROMPTS_FILE \
        --output_dir $INFERENCE_OUTPUT/$exp_name \
        --num_images 4
done

echo "Batch inference completed!"
