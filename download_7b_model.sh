# 1. 先停止当前脚本（如果还在运行）
# 按 Ctrl+C

# 2. 备份原脚本
cp download_7b_model.sh download_7b_model_backup.sh

# 3. 重新创建脚本，移除所有sudo命令
cat > download_7b_model.sh << 'EOF'
#!/bin/bash
# 下载7B基础模型脚本 - 确保使用数据盘
set -e

echo " 验证存储路径..."

# 检查数据盘路径
if [ ! -d "/root/autodl-tmp" ]; then
    echo "❌ 错误：数据盘路径 /root/autodl-tmp 不存在！"
    exit 1
fi

# 检查数据盘可用空间
DATA_DISK_SPACE=$(df /root/autodl-tmp | awk 'NR==2 {print $4}')
echo "✅ 数据盘可用空间: ${DATA_DISK_SPACE}MB"

# 确认使用数据盘路径
echo "✅ 确认使用数据盘路径: /root/autodl-tmp"
echo ""

echo " 开始下载7B基础模型..."

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kb_enterprise
echo "✅ 已激活环境: $(conda info --envs | grep '*')"

# 创建模型目录（确保在数据盘）
echo " 创建模型目录..."
mkdir -p /root/autodl-tmp/enterprise_kb/models/transformers
cd /root/autodl-tmp/enterprise_kb/models/transformers
echo "✅ 当前工作目录: $(pwd)"

# 检查Git LFS
if ! command -v git-lfs &> /dev/null; then
    echo "📥 安装Git LFS..."
    # 尝试多种安装方式
    if command -v apt-get &> /dev/null; then
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
        apt-get install git-lfs -y
    elif command -v conda &> /dev/null; then
        conda install -c conda-forge git-lfs -y
    else
        echo "❌ 无法安装Git LFS，请手动安装"
        exit 1
    fi
    git lfs install
else
    echo "✅ Git LFS已安装"
fi

# 下载模型
echo "🚀 开始下载DeepSeek-R1-Distill-Qwen-7B模型..."
echo "   模型大小: 约15-20GB"
echo "   存储位置: $(pwd)"
echo "   可用空间: $(df /root/autodl-tmp | awk 'NR==2 {print $4}')MB"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_DIR="DeepSeek-R1-Distill-Qwen-7B"

if [ -d "$MODEL_DIR" ]; then
    echo "⚠️  模型目录已存在，是否重新下载？"
    read -p "重新下载将覆盖现有文件 (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 下载已取消"
        exit 0
    fi
    rm -rf "$MODEL_DIR"
fi

echo " 克隆模型仓库到数据盘..."
git clone https://huggingface.co/$MODEL_NAME

# 验证下载
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "✅ 模型下载成功！"
    echo "📁 模型路径: $(pwd)/$MODEL_DIR"
    
    # 显示模型信息
    echo "📊 模型信息："
    ls -lh "$MODEL_DIR" | head -10
    
    # 检查关键文件
    echo "🔍 检查关键文件："
    [ -f "$MODEL_DIR/config.json" ] && echo "✅ config.json"
    [ -f "$MODEL_DIR/pytorch_model.bin" ] && echo "✅ pytorch_model.bin"
    [ -f "$MODEL_DIR/tokenizer.json" ] && echo "✅ tokenizer.json"
else
    echo "❌ 模型下载失败！"
    exit 1
fi

# 创建模型软链接（在数据盘内）
echo "🔗 创建模型软链接..."
cd /root/autodl-tmp/enterprise_kb/LLaMA-Factory
mkdir -p models
ln -sf /root/autodl-tmp/enterprise_kb/models/transformers/$MODEL_DIR ./models/
echo "✅ 模型软链接创建完成"

# 更新训练配置
echo "⚙️ 更新训练配置..."
if [ -f "configs/enterprise_kb_lora.yaml" ]; then
    sed -i "s|model_name_or_path:.*|model_name_or_path: ./models/$MODEL_DIR|" configs/enterprise_kb_lora.yaml
    echo "✅ 训练配置已更新"
else
    echo "⚠️  训练配置文件不存在，请先检查LLaMA-Factory安装"
fi

echo ""
echo "🎉 7B基础模型下载完成！"
echo ""
echo " 模型信息："
echo "   名称: DeepSeek-R1-Distill-Qwen-7B"
echo "   路径: /root/autodl-tmp/enterprise_kb/models/transformers/$MODEL_DIR"
echo "   软链接: /root/autodl-tmp/enterprise_kb/LLaMA-Factory/models/$MODEL_DIR"
echo "   存储位置: 数据盘 (/root/autodl-tmp)"
echo ""
echo "🚀 下一步操作："
echo "1. 检查模型文件完整性"
echo "2. 开始LoRA微调训练"
echo "3. 配置千问Agent集成"
EOF

# 4. 给脚本执行权限
chmod +x download_7b_model.sh

# 5. 验证脚本内容
echo "=== 检查脚本第37行 ==="
sed -n '35,40p' download_7b_model.sh

echo "=== 检查是否还有sudo ==="
grep -n "sudo" download_7b_model.sh || echo "✅ 没有找到sudo命令"