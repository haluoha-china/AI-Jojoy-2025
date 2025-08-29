#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业知识库多模态向量化系统 (修复版本)
支持文本、图像、视频、音频的统一向量化和检索
"""

import os
import sys
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# 图像处理
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# 音频处理
import librosa
import soundfile as sf

# OCR和图表理解
import easyocr
from transformers import pipeline

# 文本处理和向量化
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import faiss

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalVectorizer:
    """多模态向量化器 - 支持文本、图像、视频、音频的统一处理"""
    
    def __init__(self,
                 text_model: str = "BAAI/bge-large-zh-v1.5",
                 image_model: str = "microsoft/DialoGPT-medium",
                 device: str = "auto"):
        """
        初始化多模态向量化器
        
        Args:
            text_model: 文本嵌入模型名称
            image_model: 图像理解模型名称
            device: 计算设备 ('auto', 'cuda', 'cpu')
        """
        try:
            import torch
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = "cpu"
        self.text_model = text_model
        self.image_model = image_model
        
        # 初始化组件
        self.text_embeddings = None
        self.text_splitter = None
        self.ocr_reader = None
        self.chart_analyzer = None
        self.vector_db = None
        
        logger.info(f"初始化多模态向量化器，设备: {self.device}")
        self._initialize_processors()
    
    def _initialize_processors(self):
        """初始化所有处理器"""
        try:
            # 1. 文本嵌入模型
            logger.info("初始化文本嵌入模型...")
            self.text_embeddings = SentenceTransformer(self.text_model, device=self.device)
            
            # 2. 文本分割器
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
            )
            
            # 3. OCR文本识别
            logger.info("初始化OCR处理器...")
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                gpu_available = False
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=gpu_available)
            
            # 4. 图表理解模型
            logger.info("初始化图表理解模型...")
            self.chart_analyzer = pipeline(
                "image-to-text", 
                model=self.image_model, 
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("✅ 所有模态处理器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化处理器失败: {e}")
            raise
    
    def _fix_pil_compatibility(self, image):
        """修复PIL版本兼容性问题"""
        try:
            # 尝试使用新的resampling方法
            if hasattr(Image, 'Resampling'):
                return image.resize(image.size, Image.Resampling.LANCZOS)
            else:
                # 回退到旧版本方法
                return image.resize(image.size, Image.ANTIALIAS)
        except AttributeError:
            # 如果都不可用，使用默认方法
            return image.resize(image.size)
    
    def process_image(self, image_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        处理图像文件：OCR + 图表理解 + 图像特征
        
        Args:
            image_path: 图像文件路径
            metadata: 元数据
            
        Returns:
            文档列表
        """
        documents = []
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image.shape
            
            # 1. OCR文本提取
            logger.info(f"执行OCR识别: {image_path}")
            ocr_results = self.ocr_reader.readtext(image_path)
            
            if ocr_results:
                ocr_text = " ".join([result[1] for result in ocr_results])
                ocr_doc = Document(
                    page_content=f"图像中的文字内容: {ocr_text}",
                    metadata={
                        "source": image_path,
                        "type": "image_ocr",
                        "ocr_text": ocr_text,
                        "ocr_confidence": [result[2] for result in ocr_results],
                        **(metadata or {})
                    }
                )
                documents.append(ocr_doc)
                logger.info(f"OCR识别到 {len(ocr_results)} 个文本区域")
            
            # 2. 图表理解（尝试）
            try:
                logger.info("尝试图表理解...")
                pil_image = Image.open(image_path)
                # 修复PIL兼容性问题
                pil_image = self._fix_pil_compatibility(pil_image)
                
                chart_description = self.chart_analyzer(pil_image)[0]['generated_text']
                chart_doc = Document(
                    page_content=f"图表内容描述: {chart_description}",
                    metadata={
                        "source": image_path,
                        "type": "image_chart",
                        "chart_description": chart_description,
                        **(metadata or {})
                    }
                )
                documents.append(chart_doc)
                logger.info("图表理解完成")
                
            except Exception as e:
                logger.warning(f"图表理解失败: {e}")
            
            # 3. 图像特征描述
            # 计算基本图像特征
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 检测边缘
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            image_features = {
                "尺寸": f"{width}x{height}",
                "通道数": channels,
                "亮度": f"{brightness:.1f}",
                "对比度": f"{contrast:.1f}",
                "边缘密度": f"{edge_density:.3f}"
            }
            
            feature_doc = Document(
                page_content=f"图像特征: {json.dumps(image_features, ensure_ascii=False)}",
                metadata={
                    "source": image_path,
                    "type": "image_features",
                    "image_features": image_features,
                    **(metadata or {})
                }
            )
            documents.append(feature_doc)
            
            logger.info(f"图像处理完成: {len(documents)} 个文档")
            
        except Exception as e:
            logger.error(f"图像处理失败 {image_path}: {e}")
            # 创建错误文档
            error_doc = Document(
                page_content=f"图像处理失败: {str(e)}",
                metadata={
                    "source": image_path,
                    "type": "image_error",
                    "error": str(e),
                    **(metadata or {})
                }
            )
            documents.append(error_doc)
        
        return documents
    
    def process_video(self, video_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        处理视频文件：关键帧提取 + 音频分析
        
        Args:
            video_path: 视频文件路径
            metadata: 元数据
            
        Returns:
            文档列表
        """
        documents = []
        
        try:
            # 读取视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 1. 关键帧提取
            logger.info(f"提取视频关键帧: {video_path}")
            key_frames = self._extract_key_frames(cap, frame_count)
            
            frame_doc = Document(
                page_content=f"视频信息: {width}x{height}, {fps:.1f}fps, {duration:.1f}秒, {frame_count}帧, 提取{len(key_frames)}个关键帧",
                metadata={
                    "source": video_path,
                    "type": "video_frames",
                    "video_info": {
                        "width": width, "height": height, "fps": fps,
                        "frame_count": frame_count, "duration": duration,
                        "key_frames_count": len(key_frames)
                    },
                    **(metadata or {})
                }
            )
            documents.append(frame_doc)
            
            # 2. 音频分析（如果存在）
            try:
                audio_features = self._analyze_video_audio(video_path)
                if audio_features:
                    audio_doc = Document(
                        page_content=f"音频特征: 节拍={audio_features.get('tempo', 'N/A')}BPM, 频谱质心={audio_features.get('spectral_centroid', 'N/A')}Hz",
                        metadata={
                            "source": video_path,
                            "type": "video_audio",
                            "audio_features": audio_features,
                            **(metadata or {})
                        }
                    )
                    documents.append(audio_doc)
            except Exception as e:
                logger.warning(f"音频分析失败: {e}")
            
            cap.release()
            logger.info(f"视频处理完成: {len(documents)} 个文档")
            
        except Exception as e:
            logger.error(f"视频处理失败 {video_path}: {e}")
            error_doc = Document(
                page_content=f"视频处理失败: {str(e)}",
                metadata={
                    "source": video_path,
                    "type": "video_error",
                    "error": str(e),
                    **(metadata or {})
                }
            )
            documents.append(error_doc)
        
        return documents
    
    def _extract_key_frames(self, cap, frame_count: int, max_frames: int = 10) -> List[np.ndarray]:
        """提取关键帧"""
        key_frames = []
        if frame_count <= max_frames:
            # 如果帧数不多，全部提取
            for i in range(frame_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    key_frames.append(frame)
        else:
            # 均匀采样
            step = frame_count // max_frames
            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret and len(key_frames) < max_frames:
                    key_frames.append(frame)
        
        return key_frames
    
    def _analyze_video_audio(self, video_path: str) -> Dict[str, Any]:
        """分析视频中的音频"""
        try:
            # 使用librosa分析音频
            y, sr = librosa.load(video_path, sr=None)
            
            # 提取音频特征
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            return {
                "tempo": float(tempo),
                "spectral_centroid": float(np.mean(spectral_centroids)),
                "mfcc_mean": float(np.mean(mfccs)),
                "duration": float(len(y) / sr)
            }
        except Exception as e:
            logger.warning(f"音频分析失败: {e}")
            return {}
    
    def process_audio(self, audio_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        处理音频文件
        
        Args:
            audio_path: 音频文件路径
            metadata: 元数据
            
        Returns:
            文档列表
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # 提取音频特征
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 计算统计特征
            features = {
                "时长": f"{duration:.2f}秒",
                "采样率": f"{sr}Hz",
                "节拍": f"{tempo:.1f}BPM",
                "频谱质心": f"{np.mean(spectral_centroids):.1f}Hz",
                "MFCC均值": f"{np.mean(mfccs):.3f}",
                "音量": f"{np.mean(np.abs(y)):.3f}"
            }
            
            audio_doc = Document(
                page_content=f"音频特征: {json.dumps(features, ensure_ascii=False)}",
                metadata={
                    "source": audio_path,
                    "type": "audio",
                    "audio_features": features,
                    **(metadata or {})
                }
            )
            
            logger.info(f"音频处理完成: {audio_path}")
            return [audio_doc]
            
        except Exception as e:
            logger.error(f"音频处理失败 {audio_path}: {e}")
            error_doc = Document(
                page_content=f"音频处理失败: {str(e)}",
                metadata={
                    "source": audio_path,
                    "type": "audio_error",
                    "error": str(e),
                    **(metadata or {})
                }
            )
            return [error_doc]
    
    def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        根据文件类型自动分发到相应的处理器
        
        Args:
            file_path: 文件路径
            metadata: 元数据
            
        Returns:
            文档列表
        """
        file_path = str(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        logger.info(f"处理文件: {file_path} (类型: {file_ext})")
        
        if file_ext in ['.txt', '.md', '.json', '.csv', '.pdf']:
            # 文本文件
            return self.process_text_file(file_path, metadata)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            # 图像文件
            return self.process_image(file_path, metadata)
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            # 视频文件
            return self.process_video(file_path, metadata)
        elif file_ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
            # 音频文件
            return self.process_audio(file_path, metadata)
        else:
            logger.warning(f"不支持的文件类型: {file_ext}")
            return []
    
    def process_text_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """处理文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 文本分割
            chunks = self.text_splitter.split_text(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "type": "text",
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        **(metadata or {})
                    }
                )
                documents.append(doc)
            
            logger.info(f"文本文件处理完成: {len(documents)} 个文档块")
            return documents
            
        except Exception as e:
            logger.error(f"文本文件处理失败 {file_path}: {e}")
            return []
    
    def process_directory(self, directory_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        递归处理目录中的所有文件
        
        Args:
            directory_path: 目录路径
            metadata: 元数据
            
        Returns:
            所有文档的列表
        """
        all_documents = []
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"目录不存在: {directory_path}")
            return all_documents
        
        # 支持的文件类型
        supported_extensions = {
            '.txt', '.md', '.json', '.csv', '.pdf',  # 文本
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff',  # 图像
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',  # 视频
            '.mp3', '.wav', '.flac', '.aac', '.ogg'  # 音频
        }
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    file_metadata = {
                        "file_name": file_path.name,
                        "file_size": file_path.stat().st_size,
                        "file_path": str(file_path),
                        **(metadata or {})
                    }
                    
                    documents = self.process_file(str(file_path), file_metadata)
                    all_documents.extend(documents)
                    
                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {e}")
        
        logger.info(f"目录处理完成: {len(all_documents)} 个文档")
        return all_documents
    
    def build_vector_database(self, documents: List[Document], save_path: str = "multimodal_vector_db") -> None:
        """
        构建向量数据库
        
        Args:
            documents: 文档列表
            save_path: 保存路径
        """
        if not documents:
            logger.warning("没有文档需要向量化")
            return
        
        try:
            logger.info(f"开始构建向量数据库，文档数量: {len(documents)}")
            
            # 提取文本内容
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # 使用sentence-transformers进行文本嵌入
            embeddings = self.text_embeddings.encode(texts, show_progress_bar=True)
            
            # 创建FAISS索引
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            # 归一化向量
            faiss.normalize_L2(embeddings)
            
            # 添加向量到索引
            index.add(embeddings.astype('float32'))
            
            # 保存索引和元数据
            os.makedirs(save_path, exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(index, os.path.join(save_path, "faiss.index"))
            
            # 保存元数据
            with open(os.path.join(save_path, "metadata.pkl"), 'wb') as f:
                pickle.dump(metadatas, f)
            
            # 保存文档
            with open(os.path.join(save_path, "documents.pkl"), 'wb') as f:
                pickle.dump(documents, f)
            
            # 保存配置信息
            config = {
                "model_name": self.text_model,
                "embedding_dimension": dimension,
                "document_count": len(documents),
                "document_types": list(set(doc.metadata.get("type", "unknown") for doc in documents))
            }
            
            with open(os.path.join(save_path, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.vector_db = {
                "index": index,
                "metadatas": metadatas,
                "documents": documents,
                "config": config
            }
            
            logger.info(f"✅ 向量数据库构建完成！保存路径: {save_path}")
            logger.info(f"索引信息: {dimension}维, {len(documents)}个文档")
            
        except Exception as e:
            logger.error(f"构建向量数据库失败: {e}")
            raise
    
    def search_similar(self, query: str, top_k: int = 5, filter_type: str = None) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_type: 过滤文档类型
            
        Returns:
            搜索结果列表
        """
        if not self.vector_db:
            logger.error("向量数据库未初始化")
            return []
        
        try:
            # 查询向量化
            query_embedding = self.text_embeddings.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # 搜索
            scores, indices = self.vector_db["index"].search(
                query_embedding.astype('float32'), 
                top_k
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.vector_db["documents"]):
                    doc = self.vector_db["documents"][idx]
                    metadata = self.vector_db["metadatas"][idx]
                    
                    # 类型过滤
                    if filter_type and metadata.get("type") != filter_type:
                        continue
                    
                    result = {
                        "rank": i + 1,
                        "score": float(score),
                        "content": doc.page_content,
                        "metadata": metadata
                    }
                    results.append(result)
            
            logger.info(f"搜索完成: 查询'{query}'，返回{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        if not self.vector_db:
            return {"status": "未初始化"}
        
        return {
            "status": "已初始化",
            "config": self.vector_db["config"],
            "document_count": len(self.vector_db["documents"]),
            "index_size": self.vector_db["index"].ntotal
        }

def main():
    """主函数 - 演示多模态向量化系统"""
    print("🚀 企业知识库多模态向量化系统")
    print("=" * 60)
    
    # 初始化向量化器
    vectorizer = MultimodalVectorizer()
    
    # 示例数据路径
    sample_docs = "sample_docs"
    
    if not os.path.exists(sample_docs):
        print(f"❌ 示例数据目录不存在: {sample_docs}")
        print("请先运行 test_multimodal_system.py 创建示例数据")
        return
    
    # 处理不同类型的文档
    print("\n📁 开始处理多模态文档...")
    
    # 1. 处理文本文档
    text_docs = vectorizer.process_directory(os.path.join(sample_docs, "texts"))
    
    # 2. 处理图像文档
    image_docs = vectorizer.process_directory(os.path.join(sample_docs, "images"))
    
    # 3. 处理视频文档
    video_docs = vectorizer.process_directory(os.path.join(sample_docs, "videos"))
    
    # 4. 处理音频文档
    audio_docs = vectorizer.process_directory(os.path.join(sample_docs, "audios"))
    
    # 合并所有文档
    all_documents = text_docs + image_docs + video_docs + audio_docs
    
    print(f"\n📊 文档处理统计:")
    print(f"文本文档: {len(text_docs)} 个")
    print(f"图像文档: {len(image_docs)} 个")
    print(f"视频文档: {len(video_docs)} 个")
    print(f"音频文档: {len(audio_docs)} 个")
    print(f"总计: {len(all_documents)} 个文档")
    
    if all_documents:
        # 构建向量数据库
        print("\n🔨 构建向量数据库...")
        vectorizer.build_vector_database(all_documents, "multimodal_vector_db")
        
        # 测试搜索功能
        print("\n🔍 测试搜索功能...")
        test_queries = [
            "商业合同",
            "财务报表",
            "产品图片",
            "会议视频",
            "音频文件"
        ]
        
        for query in test_queries:
            print(f"\n查询: '{query}'")
            results = vectorizer.search_similar(query, top_k=3)
            
            for result in results:
                print(f"  [{result['rank']}] 相似度: {result['score']:.3f}")
                print(f"      类型: {result['metadata'].get('type', 'unknown')}")
                print(f"      内容: {result['content'][:100]}...")
        
        # 显示数据库信息
        print("\n📊 数据库信息:")
        db_info = vectorizer.get_database_info()
        for key, value in db_info.items():
            print(f"  {key}: {value}")
    
    print("\n🎉 多模态向量化系统演示完成！")

if __name__ == "__main__":
    main()
