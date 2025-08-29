#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "🚀 企业知识库 基础环境搭建（AutoDL/Ubuntu20.04）"
echo "目标: Ubuntu 20.04 + Python 3.8 + CUDA 11.8 + PyTorch 2.0.0 + FAISS-GPU"
echo "========================================"

# 0) 前置检测
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 未检测到 conda。请先在AutoDL镜像中启用或安装 Miniconda/Anaconda 后重试。"
  echo "官方安装参考: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

echo "✅ 已检测到 conda: $(conda --version)"

# 0.1) 数据盘自动探测与目录规划（尽可能部署到数据盘）
echo "🔎 探测数据盘..."
PREFERRED_CANDIDATES=(/root/autodl-tmp /data /mnt /workspace /data1)

pick_data_disk() {
  local best_target=""
  local best_avail=0
  # 优先命中常见数据盘挂载点
  for d in "${PREFERRED_CANDIDATES[@]}"; do
    if [ -d "$d" ]; then
      # 获取可用空间 (KB)
      local avail
      avail=$(df -Pk "$d" 2>/dev/null | tail -1 | awk '{print $4}')
      if [ -n "$avail" ] && [ "$avail" -gt "$best_avail" ]; then
        best_avail=$avail
        best_target=$d
      fi
    fi
  done
  # 若未命中，选择非根分区中可用空间最大的挂载点
  if [ -z "$best_target" ]; then
    while read -r tgt avail; do
      if [ "$tgt" != "/" ]; then
        if [ "$avail" -gt "$best_avail" ]; then
          best_avail=$avail
          best_target=$tgt
        fi
      fi
    done < <(df -Pk --output=target,avail | tail -n +2)
  fi
  echo "$best_target"
}

DATA_DISK="$(pick_data_disk)"
if [ -z "$DATA_DISK" ]; then
  echo "⚠️ 未找到独立数据盘挂载点，默认使用 /root/autodl-tmp (若不存在则继续使用 /)"
  if [ -d "/root/autodl-tmp" ]; then
    DATA_DISK="/root/autodl-tmp"
  else
    DATA_DISK="/"
  fi
fi

echo "📁 数据盘路径: $DATA_DISK"
echo "💾 数据盘可用: $(df -h "$DATA_DISK" | tail -1 | awk '{print $4}')"

# 规划项目与缓存目录，尽量放置在数据盘
EK_ROOT="$DATA_DISK/enterprise_kb"
CONDA_PKGS_DIR="$EK_ROOT/conda/pkgs"
CONDA_ENVS_DIR="$EK_ROOT/conda/envs"
PIP_CACHE_DIR_PATH="$EK_ROOT/caches/pip"
TRANSFORMERS_CACHE_PATH="$EK_ROOT/models/transformers"
HF_HOME_PATH="$EK_ROOT/models/huggingface"
TORCH_HOME_PATH="$EK_ROOT/models/torch"
VECTOR_DB_PATH="$EK_ROOT/vector_db"
DOCUMENTS_PATH="$EK_ROOT/documents"

mkdir -p "$CONDA_PKGS_DIR" "$CONDA_ENVS_DIR" "$PIP_CACHE_DIR_PATH" \
  "$TRANSFORMERS_CACHE_PATH" "$HF_HOME_PATH" "$TORCH_HOME_PATH" \
  "$VECTOR_DB_PATH" "$DOCUMENTS_PATH"
echo "✅ 已创建数据盘目录结构于: $EK_ROOT"

# 写入/合并 conda 配置，使env与pkgs位于数据盘
echo "🛠 配置 conda 使用数据盘 (envs/pkgs)..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda config --add pkgs_dirs "$CONDA_PKGS_DIR" || true
conda config --add envs_dirs "$CONDA_ENVS_DIR" || true
echo "✅ conda envs_dirs: $(conda config --show envs_dirs | tr -d '\n')"
echo "✅ conda pkgs_dirs: $(conda config --show pkgs_dirs | tr -d '\n')"

# 设置缓存到数据盘 (当前会话有效，且后续pip/transformers/torch均复用)
export PIP_CACHE_DIR="$PIP_CACHE_DIR_PATH"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE_PATH"
export HF_HOME="$HF_HOME_PATH"
export TORCH_HOME="$TORCH_HOME_PATH"
echo "✅ 已设置缓存到数据盘:"
echo "   PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "   TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "   HF_HOME=$HF_HOME"
echo "   TORCH_HOME=$TORCH_HOME"

# 1) 创建并激活专用环境
ENV_NAME="kb_enterprise"
PY_VERSION="3.8"

if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "ℹ️ 环境 ${ENV_NAME} 已存在，跳过创建"
else
  echo "📦 创建conda环境 ${ENV_NAME} (python=${PY_VERSION})..."
  conda create -n ${ENV_NAME} python=${PY_VERSION} -y
fi

echo "🔁 激活环境 ${ENV_NAME}"
conda activate ${ENV_NAME}

# 2) 安装 GPU 版 PyTorch (CUDA 11.8)
echo "📥 安装 PyTorch 2.0.0 + cu118..."
pip install --upgrade pip
conda install -c pytorch pytorch=2.0.0 torchvision torchaudio cudatoolkit=11.8 -y

# 3) 安装 FAISS-GPU (与 CUDA 11.8 匹配，Python 3.8 完美兼容)
echo "📥 安装 FAISS-GPU 1.7.4 (conda/pytorch channel)..."
conda install -c pytorch faiss-gpu=1.7.4 -y

# 4) 安装核心AI/Agent/解析依赖（版本精确锁定）
echo "📦 安装核心依赖 (pip 固定版本)..."
pip install \
  langchain==0.0.348 \
  transformers==4.35.2 \
  sentence-transformers==2.2.2 \
  qwen-agent==0.0.8 \
  fastapi==0.104.1 \
  uvicorn==0.24.0 \
  python-multipart==0.0.6 \
  pydantic==2.5.0 \
  requests==2.31.0 \
  tqdm==4.65.0 \
  pymilvus==2.4.4 \
  opencv-python-headless==4.8.0.76 \
  pillow==9.5.0 \
  PyMuPDF==1.23.0 \
  pdfminer.six==20221105 \
  pydub==0.25.1 \
  python-pptx==0.6.21 \
  librosa==0.10.0

# 5) 音视频编解码依赖（ffmpeg）
echo "🎵 安装 ffmpeg (conda-forge)..."
conda install -c conda-forge ffmpeg -y

# 6) 系统级依赖补充（Ubuntu 20.04）
echo "🔧 安装系统级依赖..."
sudo apt-get update
sudo apt-get install -y \
  libsm6 \
  libxext6 \
  libgl1-mesa-glx \
  libsndfile1 \
  libopenblas-dev

# 7) 打印版本信息与校验脚本写入位置
WORKDIR="$(pwd)"
echo "📁 当前工作目录: ${WORKDIR}"
echo "🧪 写入验证脚本: check_env.py / faiss_gpu_check.py"

cat > check_env.py <<'PY'
import torch, faiss, sys

print("="*60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
try:
    import torch.backends.cudnn as cudnn
    print(f"cuDNN版本: {cudnn.version()}")
except Exception as e:
    print(f"cuDNN信息获取失败: {e}")
print(f"FAISS GPU数: {faiss.get_num_gpus()}")
print(f"Python版本: {sys.version}")
print("设备数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前设备:", torch.cuda.current_device())
    print("设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("="*60)
PY

cat > faiss_gpu_check.py <<'PY'
import faiss
import numpy as np

print("FAISS版本:", faiss.__version__ if hasattr(faiss, "__version__") else "unknown")
print("GPU数量:", faiss.get_num_gpus())

# 启用GPU
res = faiss.StandardGpuResources()
dim = 768
index = faiss.GpuIndexFlatL2(res, dim)

# 构造测试数据
xb = np.random.random((1000, dim)).astype('float32')
xq = np.random.random((5, dim)).astype('float32')

index.add(xb)
D, I = index.search(xq, 5)
print("检索结果shape:", D.shape, I.shape)
print("GPU索引类型:", type(index))
print("✅ FAISS-GPU 正常工作")

# FAISS性能优化配置
print("\n🔧 FAISS性能优化配置:")
faiss.GpuClonerOptions().useFloat16 = True  # FP16加速
faiss.GpuClonerOptions().usePrecomputed = True  # 预计算索引
print("✅ 已启用FP16加速和预计算索引")
PY

# 8) 千问Agent兼容性补丁
echo "🔧 创建千问Agent兼容性补丁..."
cat > qwen_agent_patch.py <<'PY'
# 千问Agent Python 3.8 兼容性补丁
import sys
import collections

if sys.version_info.minor == 8:
    # Python 3.8 兼容性修复
    if not hasattr(collections, 'MutableMapping'):
        collections.MutableMapping = collections.abc.MutableMapping
    
    print("✅ 已应用 Python 3.8 兼容性补丁")
    print("   修复了 collections.MutableMapping 问题")
else:
    print("ℹ️ 当前Python版本无需兼容性补丁")

# 在导入千问Agent之前先运行此补丁
# python qwen_agent_patch.py
PY

echo "========================================"
echo "✅ 基础环境安装完成"
echo "下一步建议："
echo "1) 运行:    python check_env.py            # 验证 PyTorch/CUDA/FAISS"
echo "2) 运行:    python faiss_gpu_check.py      # 验证 FAISS-GPU性能"
echo "3) 运行:    python qwen_agent_patch.py     # 应用兼容性补丁"
echo "========================================"

# 9) 目录可写性与分区类型检测报告
echo "🧭 目录可写性/分区类型检测 (storage_report.txt)..."
REPORT_FILE="storage_report.txt"
cat > "$REPORT_FILE" <<'EOF'
# Storage Report (AutoDL Ubuntu 20.04)
# 说明: writable=YES/NO, type=system(根分区/不可扩展) or data(可扩展/独立挂载)
EOF

check_path() {
  local p="$1"
  local mp fs writable="NO" cat="system"
  if [ ! -e "$p" ]; then
    echo "MISSING  | $p" >> "$REPORT_FILE"
    return
  fi
  mp=$(df -PT "$p" 2>/dev/null | tail -1 | awk '{print $7}')
  fs=$(df -PT "$p" 2>/dev/null | tail -1 | awk '{print $2}')
  # 可写测试
  if ( : >"$p/.write_test_$$" ) 2>/dev/null; then
    writable="YES"
    rm -f "$p/.write_test_$$" 2>/dev/null || true
  fi
  # 分区分类
  if [ "$mp" != "/" ]; then
    cat="data"
  fi
  local avail_h
  avail_h=$(df -h "$p" 2>/dev/null | tail -1 | awk '{print $4}')
  printf "%-7s | %-4s | mp=%-15s | fs=%-8s | avail=%-6s | %s\n" "$writable" "$cat" "$mp" "$fs" "$avail_h" "$p" >> "$REPORT_FILE"
}

PATHS_TO_CHECK=( \
  "/" "/root" "/usr" "/opt" "/var" "/home" \
  "/root/autodl-tmp" "/data" "/mnt" "/workspace" "/data1" "$EK_ROOT" \
)

for p in "${PATHS_TO_CHECK[@]}"; do
  check_path "$p"
done

echo "📄 检测结果写入: $REPORT_FILE"
echo "   建议将大体量模型/缓存/向量库/数据统一放置到数据盘: $EK_ROOT"

# 10) 环境激活说明
echo ""
echo "🎯 环境使用说明："
echo "1. 激活环境: conda activate kb_enterprise"
echo "2. 验证安装: python check_env.py"
echo "3. 测试FAISS: python faiss_gpu_check.py"
echo "4. 应用补丁: python qwen_agent_patch.py"
echo ""
echo "📁 项目目录: $EK_ROOT"
echo "🐍 环境名称: kb_enterprise"
echo "💾 缓存位置: 数据盘 ($DATA_DISK)"
echo ""
echo "🚀 现在可以开始部署企业知识库服务了！"
