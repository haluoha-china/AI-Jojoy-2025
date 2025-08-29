"""
企业知识库 - 向量化器
实现文本向量化和FAISS索引构建
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vectorizer:
    """向量化器主类"""
    
    def __init__(self, 
                 model_name: str = "shibing624/text2vec-base-chinese",
                 device: str = "auto",
                 normalize: bool = True):
        """
        初始化向量化器
        
        Args:
            model_name: 向量模型名称
            device: 设备类型 (auto, cpu, cuda)
            normalize: 是否归一化向量
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_to_id = {}  # 文本块到ID的映射
        
        self._load_model()
    
    def _load_model(self):
        """加载向量模型"""
        try:
            logger.info(f"正在加载向量模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("向量模型加载成功")
        except Exception as e:
            logger.error(f"向量模型加载失败: {e}")
            raise
    
    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        创建文本向量嵌入
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            向量嵌入数组
        """
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"开始生成 {len(texts)} 个文本的向量嵌入")
            
            # 使用tqdm显示进度
            if show_progress:
                embeddings = []
                for text in tqdm(texts, desc="生成向量"):
                    embedding = self.model.encode([text], convert_to_tensor=False)
                    embeddings.append(embedding[0])
                embeddings = np.array(embeddings)
            else:
                embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            logger.info(f"向量嵌入生成完成，形状: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"向量嵌入生成失败: {e}")
            raise
    
    def build_index(self, chunks: List[Dict], 
                   index_type: str = "flat",
                   rebuild: bool = False) -> None:
        """
        构建FAISS索引
        
        Args:
            chunks: 文本块列表
            index_type: 索引类型 (flat, ivf, hnsw)
            rebuild: 是否重建索引
        """
        if not chunks:
            logger.warning("没有文本块，跳过索引构建")
            return
        
        if self.index is not None and not rebuild:
            logger.warning("索引已存在，如需重建请设置 rebuild=True")
            return
        
        try:
            logger.info(f"开始构建 {index_type} 类型的FAISS索引")
            
            # 保存文本块
            self.chunks = chunks
            
            # 创建文本到ID的映射
            self.chunk_to_id = {chunk['chunk_id']: i for i, chunk in enumerate(chunks)}
            
            # 提取文本内容
            texts = [chunk['content'] for chunk in chunks]
            
            # 生成向量嵌入
            embeddings = self.create_embeddings(texts)
            
            if embeddings.size == 0:
                logger.error("向量嵌入生成失败")
                return
            
            # 创建索引
            self._create_faiss_index(embeddings, index_type)
            
            logger.info(f"FAISS索引构建完成，包含 {len(chunks)} 个文档块")
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise
    
    def _create_faiss_index(self, embeddings: np.ndarray, index_type: str):
        """
        创建FAISS索引
        
        Args:
            embeddings: 向量嵌入
            index_type: 索引类型
        """
        dimension = embeddings.shape[1]
        
        if index_type == "flat":
            # 内积索引，适合小规模数据
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            # 倒排文件索引，适合中等规模数据
            nlist = min(100, len(embeddings) // 10)  # 聚类中心数量
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            # 需要训练
            self.index.train(embeddings)
        elif index_type == "hnsw":
            # 分层导航小世界图，适合大规模数据
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32是邻居数量
            self.index.hnsw.efConstruction = 200  # 构建时的搜索深度
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        # 归一化向量（如果启用）
        if self.normalize:
            faiss.normalize_L2(embeddings)
        
        # 添加向量到索引
        self.index.add(embeddings.astype('float32'))
        
        # 设置搜索参数
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10  # IVF索引的搜索深度
    
    def search(self, query: str, top_k: int = 5, 
               score_threshold: float = 0.0) -> List[Dict]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            搜索结果列表
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_index")
        
        if not query.strip():
            return []
        
        try:
            # 查询向量化
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            
            # 归一化查询向量
            if self.normalize:
                faiss.normalize_L2(query_embedding)
            
            # 搜索
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.chunks))
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # 检查索引有效性
                if idx < 0 or idx >= len(self.chunks):
                    continue
                
                # 检查相似度阈值
                if score < score_threshold:
                    continue
                
                chunk = self.chunks[idx]
                results.append({
                    'chunk': chunk,
                    'score': float(score),
                    'rank': len(results) + 1,
                    'similarity': self._score_to_similarity(score)
                })
            
            logger.info(f"搜索完成，找到 {len(results)} 个相关结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def _score_to_similarity(self, score: float) -> float:
        """
        将FAISS分数转换为相似度（0-1范围）
        
        Args:
            score: FAISS原始分数
            
        Returns:
            相似度分数
        """
        if self.normalize:
            # 归一化后的内积分数，范围在[-1, 1]之间
            return (score + 1) / 2
        else:
            # 非归一化分数，需要根据实际情况调整
            return min(max(score / 10, 0), 1)  # 假设最大分数约为10
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        批量搜索
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回结果数量
            
        Returns:
            每个查询的搜索结果列表
        """
        if not queries:
            return []
        
        try:
            # 批量向量化
            query_embeddings = self.model.encode(queries, convert_to_tensor=False)
            
            # 归一化
            if self.normalize:
                faiss.normalize_L2(query_embeddings)
            
            # 批量搜索
            scores, indices = self.index.search(
                query_embeddings.astype('float32'), 
                min(top_k, len(self.chunks))
            )
            
            all_results = []
            for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
                query_results = []
                for score, idx in zip(query_scores, query_indices):
                    if idx >= 0 and idx < len(self.chunks):
                        chunk = self.chunks[idx]
                        query_results.append({
                            'chunk': chunk,
                            'score': float(score),
                            'rank': len(query_results) + 1,
                            'similarity': self._score_to_similarity(score)
                        })
                all_results.append(query_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"批量搜索失败: {e}")
            return [[] for _ in queries]
    
    def add_chunks(self, new_chunks: List[Dict]) -> None:
        """
        添加新的文本块到索引
        
        Args:
            new_chunks: 新的文本块列表
        """
        if not new_chunks:
            return
        
        try:
            logger.info(f"添加 {len(new_chunks)} 个新文本块到索引")
            
            # 生成新文本块的向量
            new_texts = [chunk['content'] for chunk in new_chunks]
            new_embeddings = self.create_embeddings(new_texts)
            
            if new_embeddings.size == 0:
                return
            
            # 归一化新向量
            if self.normalize:
                faiss.normalize_L2(new_embeddings)
            
            # 添加到索引
            self.index.add(new_embeddings.astype('float32'))
            
            # 更新文本块列表
            start_idx = len(self.chunks)
            for i, chunk in enumerate(new_chunks):
                chunk['chunk_id'] = f"chunk_{start_idx + i}"
                self.chunks.append(chunk)
                self.chunk_to_id[chunk['chunk_id']] = start_idx + i
            
            logger.info(f"新文本块添加完成，当前索引包含 {len(self.chunks)} 个文档块")
            
        except Exception as e:
            logger.error(f"添加新文本块失败: {e}")
            raise
    
    def remove_chunks(self, chunk_ids: List[str]) -> None:
        """
        从索引中移除文本块
        
        Args:
            chunk_ids: 要移除的文本块ID列表
        """
        if not chunk_ids:
            return
        
        try:
            logger.info(f"从索引中移除 {len(chunk_ids)} 个文本块")
            
            # 注意：FAISS不支持直接删除向量，这里需要重建索引
            # 在实际应用中，建议使用支持删除的向量数据库如ChromaDB
            
            # 标记要保留的文本块
            keep_chunks = []
            for chunk in self.chunks:
                if chunk['chunk_id'] not in chunk_ids:
                    keep_chunks.append(chunk)
            
            # 重建索引
            self.build_index(keep_chunks, rebuild=True)
            
            logger.info(f"文本块移除完成，当前索引包含 {len(self.chunks)} 个文档块")
            
        except Exception as e:
            logger.error(f"移除文本块失败: {e}")
            raise
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息
        
        Returns:
            索引信息字典
        """
        if self.index is None:
            return {'status': 'not_built'}
        
        info = {
            'status': 'built',
            'total_chunks': len(self.chunks),
            'index_type': type(self.index).__name__,
            'dimension': self.index.d,
            'total_vectors': self.index.ntotal
        }
        
        # 添加特定索引类型的信息
        if hasattr(self.index, 'nlist'):
            info['nlist'] = self.index.nlist  # IVF索引的聚类中心数量
        if hasattr(self.index, 'nprobe'):
            info['nprobe'] = self.index.nprobe  # IVF索引的搜索深度
        
        return info
    
    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件
        
        Args:
            filepath: 保存文件路径
        """
        if self.index is None:
            logger.warning("索引未构建，无法保存")
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(self.index, filepath + '.faiss')
            
            # 保存文本块数据
            with open(filepath + '.chunks', 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # 保存配置信息
            config = {
                'model_name': self.model_name,
                'normalize': self.normalize,
                'chunk_to_id': self.chunk_to_id
            }
            with open(filepath + '.config', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"索引已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise
    
    def load_index(self, filepath: str) -> None:
        """
        从文件加载索引
        
        Args:
            filepath: 索引文件路径
        """
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(filepath + '.faiss')
            
            # 加载文本块数据
            with open(filepath + '.chunks', 'rb') as f:
                self.chunks = pickle.load(f)
            
            # 加载配置信息
            with open(filepath + '.config', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 恢复配置
            self.model_name = config.get('model_name', self.model_name)
            self.normalize = config.get('normalize', self.normalize)
            self.chunk_to_id = config.get('chunk_to_id', {})
            
            logger.info(f"索引加载完成，包含 {len(self.chunks)} 个文档块")
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise
    
    def export_chunks(self, filepath: str, format: str = 'json') -> None:
        """
        导出文本块数据
        
        Args:
            filepath: 导出文件路径
            format: 导出格式 (json, csv)
        """
        if not self.chunks:
            logger.warning("没有文本块数据可导出")
            return
        
        try:
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(self.chunks)
                df.to_csv(filepath, index=False, encoding='utf-8')
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"文本块数据已导出到: {filepath}")
            
        except Exception as e:
            logger.error(f"导出文本块数据失败: {e}")
            raise

# 使用示例
if __name__ == "__main__":
    # 创建向量化器
    vectorizer = Vectorizer()
    
    # 示例文本块
    sample_chunks = [
        {
            'chunk_id': 'chunk_1',
            'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
            'page_num': 1,
            'chunk_type': 'sentence'
        },
        {
            'chunk_id': 'chunk_2',
            'content': '机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。',
            'page_num': 1,
            'chunk_type': 'sentence'
        },
        {
            'chunk_id': 'chunk_3',
            'content': '深度学习是机器学习的一个分支，使用神经网络来模拟人脑的学习过程。',
            'page_num': 2,
            'chunk_type': 'sentence'
        }
    ]
    
    # 构建索引
    vectorizer.build_index(sample_chunks)
    
    # 搜索测试
    query = "什么是机器学习？"
    results = vectorizer.search(query, top_k=2)
    
    print(f"查询: {query}")
    print("搜索结果:")
    for result in results:
        print(f"- {result['chunk']['content']} (相似度: {result['similarity']:.3f})")
    
    # 获取索引信息
    info = vectorizer.get_index_info()
    print(f"\n索引信息: {info}")
    
    # 保存索引
    vectorizer.save_index("test_index")
