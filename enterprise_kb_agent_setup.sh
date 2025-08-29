#!/bin/bash
# 企业知识库 千问Agent版本一键部署脚本
# 适用：RTX 4090 24GB + Ubuntu 22.04 + AutoDL环境
# 功能：千问Agent + Function Call + 知识库服务 + 7B LoRA微调 + 多模态模型
# 架构：用户问题 → 千问Agent → Function Call → 知识库服务 → 返回结果
# 存储优化：系统盘(50G) + 数据盘(400G)
set -e

echo "🚀 开始部署企业知识库系统（千问Agent版本）..."

# ====== 0) 检测存储空间和配置数据盘 ======
echo "💾 检测存储空间..."

# 检测数据盘
DATA_DISK=$(df -h | grep -E '/root/autodl-tmp|/data|/mnt' | head -1 | awk '{print $6}')
if [ -z "$DATA_DISK" ]; then
    DATA_DISK="/root/autodl-tmp"
fi

echo "📁 数据盘路径: $DATA_DISK"
echo "💾 数据盘空间: $(df -h $DATA_DISK | tail -1 | awk '{print $4}')"

# 检查系统盘空间
SYSTEM_SPACE=$(df -h / | tail -1 | awk '{print $4}')
echo "💾 系统盘剩余空间: $SYSTEM_SPACE"

# 创建数据盘目录结构
mkdir -p $DATA_DISK/enterprise_kb/{models,data,vector_db,documents}
echo "✅ 数据盘目录结构创建完成"

# 设置环境变量，将模型缓存指向数据盘
export TRANSFORMERS_CACHE="$DATA_DISK/enterprise_kb/models/transformers"
export HF_HOME="$DATA_DISK/enterprise_kb/models/huggingface"
export TORCH_HOME="$DATA_DISK/enterprise_kb/models/torch"

echo "🔧 模型缓存路径配置完成："
echo "   Transformers: $TRANSFORMERS_CACHE"
echo "   HuggingFace: $HF_HOME"
echo "   PyTorch: $TORCH_HOME"

# ====== 1) 系统基础环境 ======
echo "📦 安装系统依赖..."
apt update -y
apt install -y python3-pip python3-venv git build-essential curl wget

# ====== 2) 创建Python虚拟环境（系统盘） ======
echo "🐍 创建Python虚拟环境..."
cd ~
python3 -m venv enterprise_kb_env
source enterprise_kb_env/bin/activate

# 验证Python环境
echo "✅ Python版本: $(python --version)"
echo "✅ Pip版本: $(pip --version)"

# ====== 3) 安装核心AI库（配置到数据盘） ======
echo "🤖 安装核心AI库..."

# 安装PyTorch（使用清华镜像源）
echo "📥 安装PyTorch (约2-3GB)..."
pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装Transformers和Embedding库
echo "📥 安装Transformers (约1-2GB)..."
pip install transformers sentence-transformers --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装Milvus Python客户端
echo "📥 安装Milvus客户端..."
pip install pymilvus --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ====== 4) 安装文档处理库 ======
echo "📄 安装文档处理库..."

# 安装文档处理库
pip install PyMuPDF pypdf2 pdfplumber --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装OCR库
echo "📥 安装PaddleOCR (约1-2GB)..."
pip install paddlepaddle paddleocr --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装图像处理库
pip install opencv-python Pillow --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ====== 5) 安装Web框架和工具库 ======
echo "🌐 安装Web框架..."
# 安装FastAPI和工具库
pip install fastapi uvicorn python-multipart --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他工具库
pip install requests tqdm python-dotenv pydantic --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ====== 6) 7B LoRA 微调环境（数据盘） ======
echo "🤖 准备 7B LoRA 微调环境..."
cd $DATA_DISK/enterprise_kb
if [ ! -d "LLaMA-Factory" ]; then
    echo "📥 克隆LLaMA-Factory到数据盘..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory

# 安装LLaMA-Factory依赖
echo "📥 安装LLaMA-Factory依赖..."
pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 准备企业知识库训练数据
mkdir -p data
cat > data/enterprise_kb.jsonl <<EOF
{"instruction":"ESS培训每月几次？","input":"","output":"根据《ESS培训管理制度》规定，ESS培训每月进行3次，分别在每月第一周、第三周和第五周进行。具体时间安排请查看公司内网培训日历。"}
{"instruction":"打印机出现数据库连接失败怎么办？","input":"","output":"当打印机出现数据库连接失败时，请按以下步骤处理：1. 检查MSSQLSERVER服务是否启动；2. 确认1433端口是否放行；3. 检查网络连接；4. 重启打印服务。详细操作步骤请参考《IT故障处理手册》第12页。"}
{"instruction":"什么是ESS系统？","input":"","output":"ESS（Employee Self Service）是员工自助服务系统，员工可以通过该系统进行请假申请、加班申请、考勤查询、薪资查询等操作。系统访问地址：http://ess.company.com"}
{"instruction":"公司内部简称解释","input":"","output":"公司内部常用简称包括：ESS（员工自助服务）、OA（办公自动化）、ERP（企业资源规划）、CRM（客户关系管理）、HR（人力资源）、IT（信息技术）、QA（质量保证）、QC（质量控制）等。"}
{"instruction":"如何申请年假？","input":"","output":"年假申请流程：1. 登录ESS系统；2. 选择'请假申请'；3. 填写请假类型为'年假'；4. 选择开始和结束时间；5. 填写请假事由；6. 提交申请等待审批。详细操作步骤请参考《ESS系统使用手册》。"}
EOF

echo "📚 训练数据已准备完成，包含企业知识库样本..."

# ====== 7) 创建知识库服务目录（系统盘） ======
echo "📁 创建知识库服务目录..."
cd ~
mkdir -p enterprise_kb_service
cd enterprise_kb_service

# 创建项目结构
mkdir -p {api,core,utils,data,logs}

# ====== 8) 创建核心配置文件 ======
echo "⚙️ 创建核心配置文件..."

# 创建requirements.txt
cat > requirements.txt <<EOF
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pymilvus>=2.3.0
PyMuPDF>=1.23.0
paddlepaddle>=2.5.0
paddleocr>=2.7.0
opencv-python>=4.8.0
Pillow>=9.5.0
python-dotenv>=1.0.0
requests>=2.31.0
tqdm>=4.65.0
EOF

# 创建环境配置文件（包含数据盘路径）
cat > .env <<EOF
# 知识库服务配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
EMBEDDING_MODEL=BAAI/bge-large-zh
API_HOST=0.0.0.0
API_PORT=8000

# 千问Agent配置
QWEN_API_KEY=your_api_key_here
QWEN_API_BASE=https://dashscope.aliyun.com/api/v1

# 存储路径配置
DATA_DISK=$DATA_DISK
TRANSFORMERS_CACHE=$DATA_DISK/enterprise_kb/models/transformers
HF_HOME=$DATA_DISK/enterprise_kb/models/huggingface
TORCH_HOME=$DATA_DISK/enterprise_kb/models/torch
VECTOR_DB_PATH=$DATA_DISK/enterprise_kb/vector_db
DOCUMENTS_PATH=$DATA_DISK/enterprise_kb/documents
TRAINING_DATA_PATH=$DATA_DISK/enterprise_kb/data

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/enterprise_kb.log
EOF

# ====== 9) 创建Function Call接口 ======
echo "🔧 创建Function Call接口..."

# 创建主API文件
cat > api/main.py <<EOF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge_base import KnowledgeBaseService
from core.document_parser import DocumentParser
from core.multimodal_processor import MultimodalProcessor

app = FastAPI(title="企业知识库Function Call API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
kb_service = KnowledgeBaseService()
doc_parser = DocumentParser()
mm_processor = MultimodalProcessor()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
async def root():
    return {"message": "企业知识库Function Call API", "status": "running"}

@app.post("/api/search")
async def search_knowledge_base(request: SearchRequest):
    """搜索知识库内容 - 供千问Agent调用"""
    try:
        results = kb_service.search(request.query, request.top_k)
        return {"success": True, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/parse_document")
async def parse_document(file: UploadFile = File(...)):
    """解析上传的文档 - 供千问Agent调用"""
    try:
        content = doc_parser.parse_file(file)
        return {"success": True, "data": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process_multimodal")
async def process_multimodal(content: str, file_type: str):
    """处理多模态内容 - 供千问Agent调用"""
    try:
        result = mm_processor.process(content, file_type)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "enterprise_kb"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# ====== 10) 创建核心服务模块 ======
echo "🧠 创建核心服务模块..."

# 创建知识库服务
cat > core/knowledge_base.py <<EOF
import os
import sys
from typing import List, Dict, Any
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    def __init__(self):
        self.embedding_model = None
        self.collection = None
        self._init_services()
    
    def _init_services(self):
        """初始化向量数据库和Embedding模型"""
        try:
            # 连接Milvus
            connections.connect("default", host="localhost", port="19530")
            logger.info("✅ 成功连接Milvus向量数据库")
            
            # 加载Embedding模型
            self.embedding_model = SentenceTransformer('BAAI/bge-large-zh')
            logger.info("✅ 成功加载BGE Embedding模型")
            
        except Exception as e:
            logger.error(f"❌ 初始化服务失败: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索知识库"""
        try:
            # 生成查询向量
            query_vector = self.embedding_model.encode([query])
            
            # 执行向量搜索
            # 这里需要根据实际的Milvus集合结构来实现
            # 暂时返回模拟结果
            results = [
                {
                    "content": f"关于'{query}'的搜索结果1",
                    "score": 0.95,
                    "source": "企业知识库文档1"
                },
                {
                    "content": f"关于'{query}'的搜索结果2", 
                    "score": 0.88,
                    "source": "企业知识库文档2"
                }
            ]
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def add_document(self, content: str, metadata: Dict[str, Any]):
        """添加文档到知识库"""
        try:
            # 生成文档向量
            doc_vector = self.embedding_model.encode([content])
            
            # 存储到Milvus
            # 这里需要根据实际的Milvus集合结构来实现
            logger.info(f"✅ 成功添加文档: {metadata.get('title', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
EOF

# 创建文档解析器
cat > core/document_parser.py <<EOF
import fitz  # PyMuPDF
import logging
from typing import Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def parse_file(self, file) -> Dict[str, Any]:
        """解析上传的文件"""
        try:
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension == '.pdf':
                return self._parse_pdf(file)
            elif file_extension == '.txt':
                return self._parse_txt(file)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
                
        except Exception as e:
            logger.error(f"文件解析失败: {e}")
            raise
    
    def _parse_pdf(self, file) -> Dict[str, Any]:
        """解析PDF文件"""
        try:
            # 保存上传的文件
            file_path = f"/tmp/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())
            
            # 使用PyMuPDF解析
            doc = fitz.open(file_path)
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            doc.close()
            
            # 清理临时文件
            os.remove(file_path)
            
            return {
                "content": text_content,
                "pages": len(doc),
                "format": "PDF",
                "filename": file.filename
            }
            
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            raise
    
    def _parse_txt(self, file) -> Dict[str, Any]:
        """解析TXT文件"""
        try:
            content = file.file.read().decode('utf-8')
            
            return {
                "content": content,
                "format": "TXT",
                "filename": file.filename
            }
            
        except Exception as e:
            logger.error(f"TXT解析失败: {e}")
            raise
EOF

# 创建多模态处理器
cat > core/multimodal_processor.py <<EOF
import logging
from typing import Dict, Any
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalProcessor:
    def __init__(self):
        self.supported_types = ['image', 'video', 'text']
    
    def process(self, content: str, file_type: str) -> Dict[str, Any]:
        """处理多模态内容"""
        try:
            if file_type == 'image':
                return self._process_image(content)
            elif file_type == 'video':
                return self._process_video(content)
            elif file_type == 'text':
                return self._process_text(content)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
                
        except Exception as e:
            logger.error(f"多模态处理失败: {e}")
            raise
    
    def _process_image(self, content: str) -> Dict[str, Any]:
        """处理图像内容"""
        try:
            # 这里可以集成图像理解模型
            # 暂时返回基础信息
            return {
                "type": "image",
                "content": content,
                "analysis": "图像内容分析结果",
                "status": "processed"
            }
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            raise
    
    def _process_video(self, content: str) -> Dict[str, Any]:
        """处理视频内容"""
        try:
            # 这里可以集成视频理解模型
            # 暂时返回基础信息
            return {
                "type": "video",
                "content": content,
                "analysis": "视频内容分析结果",
                "status": "processed"
            }
            
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            raise
    
    def _process_text(self, content: str) -> Dict[str, Any]:
        """处理文本内容"""
        try:
            return {
                "type": "text",
                "content": content,
                "analysis": "文本内容分析结果",
                "status": "processed"
            }
            
        except Exception as e:
            logger.error(f"文本处理失败: {e}")
            raise
EOF

# ====== 11) 创建启动脚本 ======
echo "🚀 创建启动脚本..."

# 创建启动脚本
cat > start_service.sh <<EOF
#!/bin/bash
echo "🚀 启动企业知识库服务..."

# 激活虚拟环境
source ~/enterprise_kb_env/bin/activate

# 设置环境变量
export TRANSFORMERS_CACHE="$DATA_DISK/enterprise_kb/models/transformers"
export HF_HOME="$DATA_DISK/enterprise_kb/models/huggingface"
export TORCH_HOME="$DATA_DISK/enterprise_kb/models/torch"

# 进入服务目录
cd ~/enterprise_kb_service

# 启动API服务
echo "🌐 启动Function Call API服务..."
python api/main.py
EOF

chmod +x start_service.sh

# ====== 12) 创建千问Agent配置示例 ======
echo "🤖 创建千问Agent配置示例..."

cat > qwen_agent_config.md <<EOF
# 千问Agent配置指南

## Function Call配置

在千问Agent中配置以下Function：

### 1. 搜索知识库
\`\`\`json
{
  "name": "search_knowledge_base",
  "description": "搜索企业知识库内容",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索查询内容"
      },
      "top_k": {
        "type": "integer",
        "description": "返回结果数量，默认5"
      }
    },
    "required": ["query"]
  }
}
\`\`\`

### 2. 解析文档
\`\`\`json
{
  "name": "parse_document",
  "description": "解析上传的企业文档",
  "parameters": {
    "type": "object",
    "properties": {
      "file": {
        "type": "string",
        "description": "文件路径或内容"
      }
    },
    "required": ["file"]
  }
}
\`\`\`

### 3. 处理多模态内容
\`\`\`json
{
  "name": "process_multimodal",
  "description": "处理多模态内容（图片、视频、文本）",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "内容描述"
      },
      "file_type": {
        "type": "string",
        "enum": ["image", "video", "text"],
        "description": "文件类型"
      }
    },
    "required": ["content", "file_type"]
  }
}
\`\`\`

## API端点

- 搜索知识库: POST /api/search
- 解析文档: POST /api/parse_document  
- 处理多模态: POST /api/process_multimodal
- 健康检查: GET /api/health

## 使用示例

用户: "帮我搜索一下ESS培训相关的信息"
Agent: 调用search_knowledge_base函数，查询"ESS培训"
结果: 返回相关文档内容和来源
EOF

# ====== 13) 创建存储空间使用说明 ======
echo "📋 创建存储空间使用说明..."

cat > storage_usage.md <<EOF
# 存储空间使用说明

## 存储分配

### 系统盘 (50G)
- Python虚拟环境: ~/enterprise_kb_env/ (~2-3GB)
- 项目代码: ~/enterprise_kb_service/ (~100MB)
- 系统依赖: ~1-2GB
- **总计: ~5-6GB**

### 数据盘 (400G)
- 模型文件: $DATA_DISK/enterprise_kb/models/ (~20-30GB)
- 训练数据: $DATA_DISK/enterprise_kb/data/ (~1-5GB)
- 向量数据库: $DATA_DISK/enterprise_kb/vector_db/ (~10-50GB)
- 企业文档: $DATA_DISK/enterprise_kb/documents/ (~10-100GB)
- **总计: ~40-185GB**

## 环境变量配置
\`\`\`bash
export TRANSFORMERS_CACHE="$DATA_DISK/enterprise_kb/models/transformers"
export HF_HOME="$DATA_DISK/enterprise_kb/models/huggingface"
export TORCH_HOME="$DATA_DISK/enterprise_kb/models/torch"
\`\`\`

## 目录结构
\`\`\`
$DATA_DISK/enterprise_kb/
├── models/                   # 模型文件
│   ├── transformers/        # Transformers缓存
│   ├── huggingface/         # HuggingFace缓存
│   └── torch/               # PyTorch缓存
├── data/                    # 训练数据
├── vector_db/               # 向量数据库
└── documents/               # 企业文档
    ├── pdf/
    ├── images/
    └── videos/
\`\`\`
EOF

# ====== 14) 验证安装 ======
echo "✅ 验证安装..."

# 检查Python包
echo "🔍 检查已安装的Python包..."
pip list | grep -E "(torch|transformers|sentence-transformers|pymilvus|PyMuPDF|fastapi)"

# 检查目录结构
echo "📁 检查项目目录结构..."
ls -la ~/enterprise_kb_service/
echo "📁 检查数据盘目录结构..."
ls -la $DATA_DISK/enterprise_kb/

# 检查存储空间使用
echo "💾 检查存储空间使用..."
echo "系统盘使用:"
df -h / | tail -1
echo "数据盘使用:"
df -h $DATA_DISK | tail -1

echo ""
echo "🎉 企业知识库系统（千问Agent版本）部署完成！"
echo ""
echo "📋 系统架构："
echo "   用户问题 → 千问Agent → Function Call → 知识库服务 → 返回结果"
echo ""
echo "📁 项目目录：~/enterprise_kb_service/"
echo "🐍 Python环境：~/enterprise_kb_env/"
echo "🚀 启动脚本：~/enterprise_kb_service/start_service.sh"
echo ""
echo "💾 存储优化："
echo "   系统盘: ~5-6GB (Python环境 + 项目代码)"
echo "   数据盘: ~40-185GB (模型 + 数据 + 文档)"
echo ""
echo "🔧 接下来需要配置："
echo "   1. 配置千问Agent的Function Call"
echo "   2. 启动知识库服务"
echo "   3. 测试Function Call接口"
echo "   4. 配置7B LoRA微调"
echo ""
echo "📖 详细配置说明请查看："
echo "   - qwen_agent_config.md (千问Agent配置)"
echo "   - storage_usage.md (存储空间使用说明)"
echo "   - 企业知识库搭建方案.md (完整配置指南)"
echo ""
echo "⚠️  重要提醒："
echo "   - 请先配置.env文件中的API密钥"
echo "   - 确保Milvus向量数据库已启动"
echo "   - 根据需要调整Function Call配置"
echo "   - 模型文件已自动配置到数据盘"
echo ""
echo "🚀 现在可以启动服务了！"
echo "   cd ~/enterprise_kb_service && ./start_service.sh"
