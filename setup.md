# 环境搭建指南

> 本文档详细说明如何搭建项目运行环境，预计耗时 \*\*5-10分钟\*\*。

\---

## 📋 系统要求

|项目|最低要求|推荐配置|
|-|-|-|
|**操作系统**|Linux (Ubuntu 20.04+) / Windows (WSL2)|Ubuntu 22.04 LTS|
|**GPU**|NVIDIA GPU with 16GB VRAM|RTX 4090 24GB / A100 40GB|
|**CUDA**|11.8|12.1|
|**Python**|3.10|3.10|
|**硬盘空间**|50GB|100GB+ (含模型缓存)|
|**内存**|16GB|32GB|

\---

## 🚀 快速开始

### 步骤1: 创建Conda环境

```bash
# 创建新环境 (Python 3.10)
conda create -n lora-safety python=3.10 -y

# 激活环境
conda activate lora-safety

# 验证Python版本
python --version  # 应显示 Python 3.10.x
```

> 💡 \*\*提示\*\*: 如果没有conda，可从 \[Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 \[Anaconda](https://www.anaconda.com/download) 安装。

\---

### 步骤2: 安装PyTorch

**根据你的CUDA版本选择对应的命令:**

#### 🔹 CUDA 12.1 (推荐)

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 🔹 CUDA 11.8

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 🔹 CPU-only (不推荐，仅用于代码测试)

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ \*\*重要\*\*: 务必确认你的CUDA版本。在终端运行 `nvidia-smi` 查看右上角显示的CUDA Version。

\---

### 步骤3: 安装项目依赖

```bash
# 进入项目根目录
cd lora-safety-project

# 安装所有依赖
pip install -r requirements.txt
```

`requirements.txt` 内容如下:

```
# ===== 核心深度学习框架 =====
torch>=2.0.0
torchvision>=0.15.0

# ===== 扩散模型与Transformer =====
diffusers>=0.27.0          # HuggingFace扩散模型库
transformers>=4.38.0       # HuggingFace Transformer模型
accelerate>=0.27.0         # 分布式训练加速

# ===== LoRA与PEFT =====
peft>=0.9.0                # 参数高效微调 (含LoRA实现)

# ===== 训练工具 =====
wandb>=0.16.0              # 实验跟踪与可视化
safetensors>=0.4.0         # 安全模型权重格式
omegaconf>=2.3.0           # YAML配置管理
einops>=0.7.0              # 张量操作工具

# ===== 图像处理 =====
Pillow>=10.0.0             # 图像读写

# ===== 数据与工具 =====
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.66.0               # 进度条

# ===== 可选加速 =====
xformers>=0.0.23           # 内存高效Attention (可选但推荐)

# ===== 评估指标 (成员D使用) =====
scipy>=1.11.0
scikit-image>=0.21.0
lpips>=0.1.4               # 感知相似度指标

# ===== 可视化 =====
matplotlib>=3.7.0
seaborn>=0.12.0
```

\---

### 步骤4: 验证安装

```bash
python -c "
import torch
print(f'✅ PyTorch版本: {torch.\_\_version\_\_}')
print(f'✅ CUDA可用: {torch.cuda.is\_available()}')
print(f'✅ CUDA版本: {torch.version.cuda}')
print(f'✅ GPU数量: {torch.cuda.device\_count()}')
if torch.cuda.is\_available():
    print(f'✅ GPU型号: {torch.cuda.get\_device\_name(0)}')
    print(f'✅ GPU显存: {torch.cuda.get\_device\_properties(0).total\_memory / 1024\*\*3:.1f} GB')
"
```

**期望输出示例:**

```
✅ PyTorch版本: 2.2.0+cu121
✅ CUDA可用: True
✅ CUDA版本: 12.1
✅ GPU数量: 1
✅ GPU型号: NVIDIA GeForce RTX 4090
✅ GPU显存: 24.0 GB
```

\---

### 步骤5: 登录HuggingFace

项目需要从HuggingFace下载预训练模型(SD 1.5等)，需要登录。

```bash
# 安装huggingface-hub (通常已包含在requirements.txt中)
pip install huggingface-hub

# 登录
huggingface-cli login
```

然后输入你的Token。获取Token步骤:

1. 访问 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. 点击 "New token"
3. 选择 "Read" 权限
4. 复制Token并粘贴到终端

> 💡 \*\*加速下载\*\*: 国内用户可设置镜像:
> ```bash
> export HF\_ENDPOINT=https://hf-mirror.com
> ```

\---

### 步骤6: 登录Weights \& Biases (可选)

用于训练过程可视化监控，强烈推荐。

```bash
wandb login
```

输入你的API Key。获取步骤:

1. 访问 [https://wandb.ai/authorize](https://wandb.ai/authorize)
2. 复制API Key并粘贴到终端

> 如果跳过此步骤，训练仍可正常运行，只是不会记录到WandB。

\---

## 🧪 功能验证

运行以下命令验证整个框架是否正常工作:

```bash
# 1. 生成配置文件
python scripts/generate\_configs.py

# 2. 检查生成的配置文件
ls configs/
# 应显示: exp01\_only\_q.yaml \~ exp08\_all.yaml

# 3. 测试框架加载 (不实际训练)
python -c "
import sys
sys.path.insert(0, 'src')
from config\_system import LoRAConfig
config = LoRAConfig.from\_yaml('configs/exp01\_only\_q.yaml')
print(f'✅ 配置加载成功: {config.experiment\_name}')
print(f'   目标层: {config.target\_modules}')
print(f'   Rank: {config.rank}')
"
```

\---

## ⚠️ 常见问题 (FAQ)

### Q1: pip安装速度很慢？

**解决方案:**

```bash
# 使用清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或永久设置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 安装时提示"CUDA版本不匹配"？

**诊断:**

```bash
nvidia-smi        # 查看CUDA版本 (右上角)
nvcc --version    # 查看CUDA编译器版本
```

**解决方案:** 确保PyTorch的CUDA版本与系统CUDA版本匹配。如果不匹配，重新安装对应版本的PyTorch。

### Q3: 运行训练时显存不足 (OOM)？

**解决方案 (按优先级):**

1. **减小batch\_size** (在配置文件中修改):

```yaml
   train\_batch\_size: 1  # 最小为1
   ```

2. **降低图像分辨率**:

```yaml
   resolution: 512  # 从768或1024降低
   ```

3. **开启梯度检查点** (默认已开启):

```yaml
   gradient\_checkpointing: true
   ```

4. **使用fp16混合精度** (默认已开启):

```yaml
   mixed\_precision: "fp16"
   ```

5. **减小LoRA rank**:

```yaml
   rank: 4  # 从8降低到4
   ```

6. **安装xformers** (可选):

```bash
   pip install xformers --no-deps
   ```

### Q4: 模型下载失败或超时？

**解决方案:**

```bash
# 方法1: 设置镜像
export HF\_ENDPOINT=https://hf-mirror.com

# 方法2: 手动下载模型
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./models/sd-v1-5

# 方法3: 使用国内镜像站
# 访问 https://hf-mirror.com 获取镜像地址
```

### Q5: xformers安装失败？

**说明:** xformers是可选依赖，不安装也能运行，只是训练速度稍慢。

**尝试安装:**

```bash
# 方法1: 直接安装
pip install xformers

# 方法2: 从源码安装 (如果预编译包不可用)
pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# 方法3: 跳过 (不影响核心功能)
# 在配置文件中设置: enable\_xformers: false
```

### Q6: Windows系统如何运行？

**推荐方案:** 使用WSL2 (Windows Subsystem for Linux)

```powershell
# 在PowerShell中安装WSL2
wsl --install

# 安装Ubuntu
wsl --install -d Ubuntu-22.04

# 然后在WSL2中按照本指南操作
```

**注意:** 原生Windows运行可能遇到路径、CUDA等问题，强烈建议使用WSL2或Linux双系统。

### Q7: 没有GPU可以运行吗？

**回答:** 可以运行代码测试，但训练速度极慢，不推荐用于实际实验。

```bash
# CPU模式运行 (仅测试)
export CUDA\_VISIBLE\_DEVICES=""
python src/train\_lora\_switchable.py --config configs/exp01\_only\_q.yaml
```

\---

## 🔧 高级配置

### 多GPU训练

如果有多张GPU，accelerate会自动处理:

```bash
# 2张GPU
accelerate launch --num\_processes 2 src/train\_lora\_switchable.py --config configs/exp01\_only\_q.yaml
```

### 自定义CUDA路径

如果CUDA安装在非默认位置:

```bash
export CUDA\_HOME=/usr/local/cuda-12.1
export PATH=$CUDA\_HOME/bin:$PATH
export LD\_LIBRARY\_PATH=$CUDA\_HOME/lib64:$LD\_LIBRARY\_PATH
```

\---

## 📁 环境文件清单

搭建完成后，项目目录应包含:

```
lora-safety-project/
├── requirements.txt          # ✅ 依赖清单
├── configs/                  # ✅ 实验配置 (运行generate\_configs.py生成)
├── src/                      # ✅ 源代码
├── scripts/                  # ✅ 运行脚本
├── docs/                     # ✅ 文档
├── outputs/                  # 📁 训练输出 (运行时生成)
├── logs/                     # 📁 日志 (运行时生成)
└── ...
```

\---

## ✅ 环境检查清单

搭建完成后，请确认以下检查项:

* \[ ] Conda环境已创建并激活 (`conda activate lora-safety`)
* \[ ] PyTorch已安装且CUDA可用 (`torch.cuda.is\_available()` 返回True)
* \[ ] 所有依赖已安装 (`pip list` 显示requirements.txt中的包)
* \[ ] HuggingFace已登录 (`huggingface-cli whoami` 显示用户名)
* \[ ] 配置文件已生成 (`configs/` 目录有8个yaml文件)
* \[ ] 框架测试通过 (运行"功能验证"步骤无报错)

\---

## 📞 遇到其他问题？

1. 查看训练日志: `logs/scheduler/experiment\_schedule.log`
2. 检查显存使用: `nvidia-smi` (每2秒刷新: `watch -n 2 nvidia-smi`)
3. 查看WandB记录: [https://wandb.ai](https://wandb.ai)
4. 联系项目成员A获取支持

\---

> \*\*文档版本\*\*: v1.0  
> \*\*更新日期\*\*: 2026-05-09  
> \*\*适用项目\*\*: 文生图扩散模型安全对齐研究

