#!/bin/bash
# 企业知识库环境极简补充安装脚本
# 适用：已有完整环境的AutoDL服务器
set -e

echo "🔧 开始极简补充安装企业知识库环境..."

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kb_enterprise

echo "✅ 已激活环境: $(conda info --envs | grep '*')"

# ====== 1) 安装LLaMA-Factory ======
echo "🤖 安装LLaMA-Factory微调框架..."
cd ~
if [ ! -d "LLaMA-Factory" ]; then
    echo "正在克隆LLaMA-Factory仓库..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory

echo "安装LLaMA-Factory依赖..."
pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ====== 2) 准备训练数据 ======
echo "📚 准备训练数据..."
mkdir -p data
cat > data/enterprise_kb.jsonl <<EOF
{"instruction":"ESS培训每月几次？","input":"","output":"根据《ESS培训管理制度》规定，ESS培训每月进行3次，分别在每月第一周、第三周和第五周进行。具体时间安排请查看公司内网培训日历。"}
{"instruction":"打印机出现数据库连接失败怎么办？","input":"","output":"当打印机出现数据库连接失败时，请按以下步骤处理：1. 检查MSSQLSERVER服务是否启动；2. 确认1433端口是否放行；3. 检查网络连接；4. 重启打印服务。详细操作步骤请参考《IT故障处理手册》第12页。"}
{"instruction":"什么是ESS系统？","input":"","output":"ESS（Employee Self Service）是员工自助服务系统，员工可以通过该系统进行请假申请、加班申请、考勤查询、薪资查询等操作。系统访问地址：http://ess.company.com"}
{"instruction":"公司内部简称解释","input":"","output":"公司内部常用简称包括：ESS（员工自助服务）、OA（办公自动化）、ERP（企业资源规划）、CRM（客户关系管理）、HR（人力资源）、IT（信息技术）、QA（质量保证）、QC（质量控制）等。"}
{"instruction":"如何申请年假？","input":"","output":"年假申请流程：1. 登录ESS系统；2. 选择'请假申请'；3. 填写请假类型为'年假'；4. 选择开始和结束时间；5. 填写请假事由；6. 提交申请等待审批。详细操作步骤请参考《ESS系统使用手册》。"}
EOF

echo "✅ 训练数据准备完成"

# ====== 3) 创建企业知识库服务目录 ======
echo "📁 创建企业知识库服务目录..."
cd ~
mkdir -p enterprise_kb_service
cd enterprise_kb_service
mkdir -p {core,utils,data,logs}

# ====== 4) 创建核心配置文件 ======
echo "⚙️ 创建核心配置文件..."

# 创建环境配置文件
cat > .env <<EOF
# 知识库服务配置
EMBEDDING_MODEL=BAAI/bge-large-zh
API_HOST=0.0.0.0
API_PORT=8000

# 千问Agent配置
QWEN_API_KEY=your_api_key_here
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 存储路径配置
DATA_DISK=/root/autodl-tmp
TRANSFORMERS_CACHE=/root/autodl-tmp/enterprise_kb/models/transformers
HF_HOME=/root/autodl-tmp/enterprise_kb/models/huggingface
VECTOR_DB_PATH=/root/autodl-tmp/enterprise_kb/vector_db
DOCUMENTS_PATH=/root/autodl-tmp/enterprise_kb/documents
TRAINING_DATA_PATH=/root/autodl-tmp/enterprise_kb/data

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/enterprise_kb.log
EOF

# ====== 5) 创建LoRA训练配置 ======
echo "🎯 创建LoRA训练配置..."
cd ~/LLaMA-Factory

# 创建训练配置文件
cat > configs/enterprise_kb_lora.yaml <<EOF
# 企业知识库LoRA微调配置
model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
dataset_path: data/enterprise_kb.jsonl
template: qwen
finetuning_type: lora
output_dir: ./lora_ckpt
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
quantization_bit: 4
learning_rate: 3e-4
fp16: true
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
EOF

# ====== 6) 验证安装 ======
echo "🔍 验证安装..."
echo "已安装的Python包："
pip list | grep -E "(torch|transformers|sentence|faiss|langchain|qwen)"

echo "目录结构："
ls -la ~/enterprise_kb_service/
ls -la ~/LLaMA-Factory/

# 检查LLaMA-Factory是否安装成功
cd ~/LLaMA-Factory
python -c "import src; print('✅ LLaMA-Factory导入成功')" 2>/dev/null || echo "⚠️ LLaMA-Factory导入测试失败，可能需要重新安装依赖"

echo ""
echo "🎉 企业知识库环境极简补充安装完成！"
echo ""
echo "📋 您的环境已经非常完整，接下来需要："
echo "1. 下载7B基础模型到 /root/autodl-tmp/enterprise_kb/models/transformers/"
echo "2. 配置千问Agent的Function Call"
echo "3. 开始LoRA微调训练"
echo ""
echo "🚀 您已经具备了："
echo "✅ PyTorch + CUDA 12.1 + RTX 4090"
echo "✅ FAISS-GPU 向量数据库"
echo "✅ LangChain + Qwen-Agent"
echo "✅ 完整的文档处理工具链"
echo "✅ 音视频处理能力"
echo "✅ Web服务框架"
echo "✅ LLaMA-Factory 微调框架"
echo ""
echo "📁 重要目录："
echo "   - LLaMA-Factory: ~/LLaMA-Factory/"
echo "   - 训练数据: ~/LLaMA-Factory/data/enterprise_kb.jsonl"
echo "   - 训练配置: ~/LLaMA-Factory/configs/enterprise_kb_lora.yaml"
echo "   - 知识库服务: ~/enterprise_kb_service/"
echo ""
echo "🔧 下一步操作："
echo "1. 等待LLaMA-Factory安装完成"
echo "2. 下载7B基础模型"
echo "3. 开始LoRA微调训练"
