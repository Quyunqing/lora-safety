# 自动顺序运行8组实验（或并行如果GPU够）

CONFIG_DIR="configs"
OUTPUT_BASE="outputs"
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 定义实验顺序
EXPERIMENTS=(
    "exp01_only_q"
    "exp02_only_k" 
    "exp03_only_v"
    "exp04_qkv"
    "exp05_resnet"
    "exp06_skip"
    "exp07_full_attn"
    "exp08_all"
)

echo "Starting all experiments at $(date)" > $LOG_DIR/experiment_schedule.log

for exp in "${EXPERIMENTS[@]}"; do
    config_file="$CONFIG_DIR/${exp}.yaml"
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config $config_file not found!" >> $LOG_DIR/experiment_schedule.log
        continue
    fi
    
    echo "========================================" >> $LOG_DIR/experiment_schedule.log
    echo "Starting $exp at $(date)" >> $LOG_DIR/experiment_schedule.log
    
    # 运行训练，后台执行并记录日志
    python train_lora_switchable.py --config $config_file \
        > $LOG_DIR/${exp}_train.log 2>&1
    
    echo "Finished $exp at $(date)" >> $LOG_DIR/experiment_schedule.log
    echo "" >> $LOG_DIR/experiment_schedule.log
    
    # 可选：实验间等待，让GPU降温
    sleep 30
done

echo "All experiments completed at $(date)" >> $LOG_DIR/experiment_schedule.log
