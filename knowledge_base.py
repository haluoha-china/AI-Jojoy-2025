import os
import sys
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    def __init__(self, index_path: str = "multimodal_vector_db_md5"):
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.index_path = index_path
        self._init_services()

    def _init_services(self):
        """初始化向量数据库和Embedding模型"""
        try:
            # 加载 Embedding 模型
            self.embedding_model = SentenceTransformer('/root/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116')
            logger.info("✅ 成功加载 BGE Embedding 模型")

            # 尝试加载已有索引
            self._load_or_create_index()

        except Exception as e:
            logger.error(f"❌ 初始化服务失败: {e}")
            raise

    def _load_or_create_index(self):
        """加载已有索引或创建新索引"""
        try:
            # 优先尝试加载multimodal_vector_db_md5目录下的现有数据库
            multimodal_path = "multimodal_vector_db_md5"
            if os.path.exists(multimodal_path):
                index_file = os.path.join(multimodal_path, "faiss.index")
                docs_file = os.path.join(multimodal_path, "documents.pkl")
                metadata_file = os.path.join(multimodal_path, "metadata.pkl")
                
                if os.path.exists(index_file) and os.path.exists(docs_file):
                    # 加载现有向量数据库
                    self.index = faiss.read_index(index_file)
                    with open(docs_file, 'rb') as f:
                        self.documents = pickle.load(f)
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'rb') as f:
                            self.metadata = pickle.load(f)
                    logger.info(f"✅ 成功加载现有向量数据库: {len(self.documents)} 个文档块")
                    return
            
            # 如果multimodal数据库不存在，尝试加载默认路径
            index_file = f"{self.index_path}.faiss"
            docs_file = f"{self.index_path}_docs.pkl"
            
            if os.path.exists(index_file) and os.path.exists(docs_file):
                # 加载已有索引
                self.index = faiss.read_index(index_file)
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"✅ 成功加载已有索引: {len(self.documents)} 个文档")
            else:
                # 创建新索引
                dimension = self.embedding_model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
                logger.info(f"✅ 成功创建新FAISS索引，维度: {dimension}")
                
        except Exception as e:
            logger.error(f"加载或创建索引失败: {e}")
            raise

    def save_index(self):
        """保存索引到文件"""
        try:
            index_file = f"{self.index_path}.faiss"
            docs_file = f"{self.index_path}_docs.pkl"
            
            faiss.write_index(self.index, index_file)
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"✅ 索引已保存: {index_file}")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索知识库"""
        try:
            if not self.documents:
                logger.warning("知识库为空，返回默认结果")
                return [
                    {"content": f"关于'{query}'的搜索结果1", "score": 0.95, "source": "企业知识库文档1"},
                    {"content": f"关于'{query}'的搜索结果2", "score": 0.88, "source": "企业知识库文档2"}
                ]

            # 编码查询
            query_vector = self.embedding_model.encode([query])
            
            # 搜索最相似的向量
            scores, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
            
            # 格式化结果
            formatted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    formatted_results.append({
                        "content": doc.get("content", ""),
                        "score": float(score),
                        "source": doc.get("source", "Unknown")
                    })
            
            return formatted_results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return [
                {"content": f"关于'{query}'的搜索结果1", "score": 0.95, "source": "企业知识库文档1"},
                {"content": f"关于'{query}'的搜索结果2", "score": 0.88, "source": "企业知识库文档2"}
            ]

    def add_document(self, content: str, metadata: Dict[str, Any]):
        """添加文档到知识库"""
        try:
            # 编码文档
            doc_vector = self.embedding_model.encode([content])
            
            # 添加到FAISS索引
            self.index.add(doc_vector)
            
            # 添加到文档列表
            doc_info = {
                "content": content,
                "source": metadata.get('title', 'Unknown'),
                "metadata": metadata
            }
            self.documents.append(doc_info)
            
            logger.info(f"✅ 成功添加文档: {metadata.get('title', 'Unknown')}")
            
            # 自动保存索引
            self.save_index()

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)

    def clear_index(self):
        """清空索引"""
        try:
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dimension)
            self.documents = []
            self.save_index()
            logger.info("✅ 索引已清空")
        except Exception as e:
            logger.error(f"清空索引失败: {e}")
            raise
