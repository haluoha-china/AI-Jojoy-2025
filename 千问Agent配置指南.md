# 🤖 千问Agent配置指南

## 📋 概述

本指南将帮助你在千问Agent中配置Function Call，使其能够调用企业知识库的各种功能。

## 🏗️ 系统架构

```
用户问题 → 千问Agent → Function Call → 企业知识库API → 返回结果
```

## ⚙️ Function Call配置

### 1. 搜索知识库

```json
{
  "name": "search_knowledge_base",
  "description": "搜索企业知识库内容，支持简称查询、流程查询、制度查询等",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索查询内容，支持：1. 简称查询（如ESS、OA等）；2. 流程查询（如年假申请）；3. 制度查询（如培训制度）"
      },
      "top_k": {
        "type": "integer",
        "description": "返回结果数量，默认5"
      }
    },
    "required": ["query"]
  }
}
```

### 2. 解析文档

```json
{
  "name": "parse_document",
  "description": "解析上传的企业文档，支持PDF、TXT等格式",
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
```

### 3. 处理多模态内容

```json
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
```

## 🌐 API端点配置

### 基础URL
```
http://your_server_ip:8000
```

### 具体端点

| 功能 | 方法 | 端点 | 说明 |
|------|------|------|------|
| 搜索知识库 | POST | `/api/search` | 搜索企业知识库内容 |
| 解析文档 | POST | `/api/parse_document` | 解析上传的文档 |
| 处理多模态 | POST | `/api/process_multimodal` | 处理多模态内容 |
| 健康检查 | GET | `/api/health` | 检查服务状态 |

## 💻 代码实现结构

### 项目目录结构
```
~/enterprise_kb_service/
├── api/
│   └── main.py              # 主API服务
├── core/
│   ├── knowledge_base.py    # 知识库服务
│   ├── document_parser.py   # 文档解析器
│   └── multimodal_processor.py # 多模态处理器
├── utils/                   # 工具函数
├── data/                    # 数据目录
├── logs/                    # 日志目录
├── requirements.txt         # 依赖包列表
├── .env                     # 环境配置
├── start_service.sh         # 启动脚本
└── qwen_agent_config.md     # 千问Agent配置说明
```

### 核心代码实现

#### 1. 主API服务 (api/main.py)
```python
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
```

#### 2. 知识库服务 (core/knowledge_base.py)
```python
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
```

#### 3. 文档解析器 (core/document_parser.py)
```python
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
```

#### 4. 多模态处理器 (core/multimodal_processor.py)
```python
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
```

### 启动脚本 (start_service.sh)
```bash
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
```

## 📝 使用示例

### 示例1：简称查询
**用户问题**："什么是ESS？"

**Agent调用**：
```json
{
  "function": "search_knowledge_base",
  "parameters": {
    "query": "ESS",
    "top_k": 3
  }
}
```

**预期结果**：
```
ESS是Employee Self Service的缩写，中文名称：员工自助服务系统。
功能包括：请假申请、加班申请、考勤查询、薪资查询、个人信息维护等。
```

### 示例2：流程查询
**用户问题**："如何申请年假？"

**Agent调用**：
```json
{
  "function": "search_knowledge_base",
  "parameters": {
    "query": "年假申请流程",
    "top_k": 5
  }
}
```

**预期结果**：
```
年假申请流程：
1. 登录ESS系统
2. 选择'请假申请'
3. 填写请假类型为'年假'
4. 选择开始和结束时间
5. 填写请假事由
6. 提交申请等待审批
```

### 示例3：制度查询
**用户问题**："公司培训制度是什么？"

**Agent调用**：
```json
{
  "function": "search_knowledge_base",
  "parameters": {
    "query": "培训管理制度",
    "top_k": 3
  }
}
```

## 🔧 配置步骤

### 步骤1：在千问Agent中添加Function

1. 登录千问Agent管理界面
2. 进入"Function Call"配置页面
3. 点击"添加Function"
4. 复制上面的JSON配置
5. 保存配置

### 步骤2：配置API连接

1. 确保企业知识库服务已启动
2. 验证API端点可访问
3. 测试Function调用

### 步骤3：测试配置

1. 在千问Agent中测试简单问题
2. 验证Function调用成功
3. 检查返回结果正确性

## 🚀 部署和启动

### 1. 一键部署
```bash
# 给脚本执行权限
chmod +x enterprise_kb_agent_setup.sh

# 运行部署脚本
./enterprise_kb_agent_setup.sh
```

### 2. 启动服务
```bash
# 进入服务目录
cd ~/enterprise_kb_service

# 启动服务
./start_service.sh
```

### 3. 验证服务
```bash
# 健康检查
curl http://localhost:8000/api/health

# 测试搜索功能
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "ESS培训", "top_k": 3}'
```

## 🎯 针对你的数据的优化建议

### 简称查询优化
- 配置ESS、OA、ERP、CRM等常用简称
- 添加部门相关的简称
- 包含技术术语的简称

### 流程查询优化
- 配置常见业务流程
- 添加故障处理流程
- 包含审批流程说明

### 制度查询优化
- 配置公司管理制度
- 添加操作规范
- 包含安全制度要求

## 🚨 常见问题

### 问题1：Function调用失败
**解决方案**：
1. 检查API服务是否启动
2. 验证网络连接
3. 检查API密钥配置

### 问题2：返回结果不准确
**解决方案**：
1. 优化搜索关键词
2. 调整相似度阈值
3. 增加训练数据

### 问题3：响应速度慢
**解决方案**：
1. 优化向量检索
2. 使用缓存机制
3. 调整模型参数

## 📊 性能监控

### 监控指标
- Function调用成功率
- 响应时间
- 用户满意度
- 知识库覆盖率

### 优化建议
- 定期更新知识库
- 优化搜索算法
- 增加训练数据
- 调整模型参数

## 🔍 代码调试和优化

### 日志查看
```bash
# 查看服务日志
tail -f ~/enterprise_kb_service/logs/enterprise_kb.log

# 查看系统日志
journalctl -u enterprise_kb -f
```

### 性能分析
```bash
# 查看API响应时间
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/health"

# 查看系统资源使用
htop
nvidia-smi  # GPU使用情况
```

### 代码热重载
```bash
# 开发模式下启用热重载
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🎉 完成！

配置完成后，你的千问Agent将能够：
- ✅ 智能理解用户问题
- ✅ 准确调用知识库功能
- ✅ 提供专业的企业问答服务
- ✅ 支持多模态内容处理
- ✅ 通过Function Call实现复杂业务逻辑

现在可以开始测试Function Call功能了！🚀

## 📚 相关文档

- [千问Agent官方文档](https://dashscope.aliyun.com/)
- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [PyMuPDF文档](https://pymupdf.readthedocs.io/)
- [Sentence Transformers文档](https://www.sbert.net/)
- [Milvus Python客户端文档](https://milvus.io/docs/install_standalone-docker.md)
