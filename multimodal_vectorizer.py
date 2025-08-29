#!/usr/bin/env python3
"""
企业知识库多模态向量化系统
支持文本、图像、视频、音频的智能向量化和检索
"""

import os
import json
import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# 文本处理
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 音频处理
import librosa
import soundfile as sf

# 图表理解
import easyocr
from transformers import pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalVectorizer:
    """多模态向量化器 - 支持文本、图像、视频、音频"""
    
    def __init__(self, 
                 text_model: str = "BAAI/bge-large-zh-v1.5",
                 image_model: str = "microsoft/DialoGPT-medium",
                 device: str = "auto"):
        """
        初始化多模态向量化器
        
        Args:
            text_model: 文本嵌入模型
            image_model: 图像理解模型
            device: 计算设备
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = text_model
        self.image_model = image_model
        
        # 初始化各模态处理器
        self.text_embeddings = None
        self.image_processor = None
        self.ocr_reader = None
        self.chart_analyzer = None
        self.audio_processor = None
        
        # 向量数据库
        self.vector_db = None
        self.text_splitter = None
        
        logger.info(f"初始化多模态向量化器，设备: {self.device}")
        self._initialize_processors()
    
    def _initialize_processors(self):
        """初始化各模态处理器"""
        try:
            # 1. 文本嵌入模型
            logger.info("加载文本嵌入模型...")
            self.text_embeddings = SentenceTransformer(
                self.text_model,
                device=self.device
            )
            
            # 2. 文本分割器
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
            
            # 3. OCR文本识别
            logger.info("初始化OCR处理器...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())
            
            # 4. 图表理解模型
            logger.info("初始化图表理解模型...")
            self.chart_analyzer = pipeline(
                "image-to-text",
                model="microsoft/DialoGPT-medium",
                device=0 if self.device == "cuda" else -1
            )
            
            # 5. 音频处理器
            logger.info("初始化音频处理器...")
            # 音频处理使用librosa，无需额外模型
            
            logger.info("✅ 所有模态处理器初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 处理器初始化失败: {e}")
            raise
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """处理文本内容"""
        try:
            # 分割长文本
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'type': 'text',
                        'chunk_id': i,
                        'original_length': len(text),
                        **(metadata or {})
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"文本处理失败: {e}")
            return []
    
    def process_image(self, image_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """处理图像内容 - OCR + 图表理解"""
        try:
            logger.info(f"处理图像: {image_path}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            documents = []
            
            # 1. OCR文本提取
            logger.info("执行OCR文本识别...")
            ocr_results = self.ocr_reader.readtext(image_path)
            
            if ocr_results:
                # 提取所有文本
                extracted_text = " ".join([result[1] for result in ocr_results])
                
                # 创建OCR文档
                ocr_doc = Document(
                    page_content=f"图像中的文本内容：{extracted_text}",
                    metadata={
                        'type': 'image_ocr',
                        'source': image_path,
                        'ocr_confidence': np.mean([result[2] for result in ocr_results]),
                        'text_count': len(ocr_results),
                        **(metadata or {})
                    }
                )
                documents.append(ocr_doc)
                
                # 如果OCR文本较长，进行分割
                if len(extracted_text) > 200:
                    text_chunks = self.text_splitter.split_text(extracted_text)
                    for i, chunk in enumerate(text_chunks):
                        chunk_doc = Document(
                            page_content=chunk,
                            metadata={
                                'type': 'image_ocr_chunk',
                                'source': image_path,
                                'chunk_id': i,
                                'ocr_confidence': np.mean([result[2] for result in ocr_results]),
                                **(metadata or {})
                            }
                        )
                        documents.append(chunk_doc)
            
            # 2. 图表理解（尝试）
            try:
                logger.info("尝试图表理解...")
                # 使用PIL加载图像
                pil_image = Image.open(image_path)
                
                # 图表理解
                chart_description = self.chart_analyzer(pil_image)[0]['generated_text']
                
                if chart_description and len(chart_description) > 10:
                    chart_doc = Document(
                        page_content=f"图表分析结果：{chart_description}",
                        metadata={
                            'type': 'image_chart',
                            'source': image_path,
                            'chart_analysis': True,
                            **(metadata or {})
                        }
                    )
                    documents.append(chart_doc)
                    
            except Exception as chart_e:
                logger.warning(f"图表理解失败: {chart_e}")
            
            # 3. 图像特征描述
            # 分析图像基本信息
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            # 计算图像统计信息
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            image_info_doc = Document(
                page_content=f"图像信息：尺寸{width}x{height}，通道数{channels}，亮度{brightness:.1f}，对比度{contrast:.1f}",
                metadata={
                    'type': 'image_info',
                    'source': image_path,
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'brightness': float(brightness),
                    'contrast': float(contrast),
                    **(metadata or {})
                }
            )
            documents.append(image_info_doc)
            
            logger.info(f"✅ 图像处理完成，生成 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"❌ 图像处理失败: {e}")
            return []
    
    def process_video(self, video_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """处理视频内容 - 关键帧提取 + 音频分析"""
        try:
            logger.info(f"处理视频: {video_path}")
            
            documents = []
            
            # 1. 视频基本信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 视频信息文档
            video_info_doc = Document(
                page_content=f"视频信息：时长{duration:.1f}秒，帧率{fps:.1f}fps，分辨率{width}x{height}，总帧数{frame_count}",
                metadata={
                    'type': 'video_info',
                    'source': video_path,
                    'duration': duration,
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height,
                    **(metadata or {})
                }
            )
            documents.append(video_info_doc)
            
            # 2. 关键帧提取和分析
            logger.info("提取关键帧...")
            key_frames = self._extract_key_frames(cap, max_frames=10)
            
            for i, frame in enumerate(key_frames):
                # 保存关键帧
                frame_path = f"{video_path}_frame_{i}.jpg"
                cv2.imwrite(frame_path, frame)
                
                # 处理关键帧
                frame_docs = self.process_image(frame_path, {
                    'type': 'video_keyframe',
                    'video_source': video_path,
                    'frame_index': i,
                    'timestamp': i * (duration / len(key_frames)),
                    **(metadata or {})
                })
                
                documents.extend(frame_docs)
                
                # 清理临时文件
                os.remove(frame_path)
            
            # 3. 音频分析（如果视频有音频轨道）
            try:
                logger.info("分析视频音频...")
                audio_docs = self._analyze_video_audio(video_path, metadata)
                documents.extend(audio_docs)
            except Exception as audio_e:
                logger.warning(f"视频音频分析失败: {audio_e}")
            
            cap.release()
            
            logger.info(f"✅ 视频处理完成，生成 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"❌ 视频处理失败: {e}")
            return []
    
    def _extract_key_frames(self, cap: cv2.VideoCapture, max_frames: int = 10) -> List[np.ndarray]:
        """提取视频关键帧"""
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= max_frames:
            # 如果帧数不多，全部提取
            for i in range(frame_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        else:
            # 均匀采样关键帧
            step = frame_count // max_frames
            for i in range(0, frame_count, step):
                if len(frames) >= max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        
        return frames
    
    def _analyze_video_audio(self, video_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """分析视频音频内容"""
        try:
            # 使用librosa分析音频
            y, sr = librosa.load(video_path, sr=None)
            
            # 音频特征
            duration = librosa.get_duration(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 创建音频分析文档
            audio_doc = Document(
                page_content=f"音频分析：时长{duration:.1f}秒，节拍{tempo:.1f}BPM，频谱质心{np.mean(spectral_centroids):.1f}Hz",
                metadata={
                    'type': 'video_audio',
                    'source': video_path,
                    'duration': duration,
                    'tempo': tempo,
                    'sample_rate': sr,
                    'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                    'mfcc_features': mfccs.shape,
                    **(metadata or {})
                }
            )
            
            return [audio_doc]
            
        except Exception as e:
            logger.warning(f"音频分析失败: {e}")
            return []
    
    def process_audio(self, audio_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """处理音频内容"""
        try:
            logger.info(f"处理音频: {audio_path}")
            
            documents = []
            
            # 1. 音频基本信息
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 音频信息文档
            audio_info_doc = Document(
                page_content=f"音频信息：时长{duration:.1f}秒，采样率{sr}Hz，文件大小{os.path.getsize(audio_path)}字节",
                metadata={
                    'type': 'audio_info',
                    'source': audio_path,
                    'duration': duration,
                    'sample_rate': sr,
                    'file_size': os.path.getsize(audio_path),
                    **(metadata or {})
                }
            )
            documents.append(audio_info_doc)
            
            # 2. 音频特征分析
            # 节拍检测
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 音频特征文档
            features_doc = Document(
                page_content=f"音频特征：节拍{tempo:.1f}BPM，频谱质心{np.mean(spectral_centroids):.1f}Hz，频谱滚降{np.mean(spectral_rolloff):.1f}Hz",
                metadata={
                    'type': 'audio_features',
                    'source': audio_path,
                    'tempo': tempo,
                    'beat_count': len(beats),
                    'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                    'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                    'mfcc_shape': mfccs.shape,
                    **(metadata or {})
                }
            )
            documents.append(features_doc)
            
            # 3. 语音转文本（如果有语音内容）
            try:
                # 这里可以集成语音识别模型
                # 暂时跳过，因为需要额外的语音识别模型
                pass
            except Exception as stt_e:
                logger.debug(f"语音转文本跳过: {stt_e}")
            
            logger.info(f"✅ 音频处理完成，生成 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"❌ 音频处理失败: {e}")
            return []
    
    def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """根据文件类型自动选择处理器"""
        try:
            file_path = str(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.txt', '.md', '.json', '.csv']:
                # 文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self.process_text(content, metadata)
                
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # 图像文件
                return self.process_image(file_path, metadata)
                
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                # 视频文件
                return self.process_video(file_path, metadata)
                
            elif file_ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                # 音频文件
                return self.process_audio(file_path, metadata)
                
            elif file_ext in ['.pdf']:
                # PDF文件（需要pdfminer）
                try:
                    from pdfminer.high_level import extract_text
                    content = extract_text(file_path)
                    return self.process_text(content, metadata)
                except ImportError:
                    logger.warning("pdfminer未安装，跳过PDF处理")
                    return []
                
            else:
                logger.warning(f"不支持的文件类型: {file_ext}")
                return []
                
        except Exception as e:
            logger.error(f"文件处理失败 {file_path}: {e}")
            return []
    
    def process_directory(self, directory: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """处理整个目录的文件"""
        try:
            logger.info(f"处理目录: {directory}")
            
            if not os.path.exists(directory):
                logger.warning(f"目录不存在: {directory}")
                return []
            
            all_documents = []
            
            # 遍历目录
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # 处理文件
                    file_docs = self.process_file(file_path, {
                        'directory': directory,
                        'relative_path': os.path.relpath(file_path, directory),
                        **(metadata or {})
                    })
                    
                    all_documents.extend(file_docs)
            
            logger.info(f"✅ 目录处理完成，生成 {len(all_documents)} 个文档")
            return all_documents
            
        except Exception as e:
            logger.error(f"❌ 目录处理失败: {e}")
            return []
    
    def build_vector_database(self, documents: List[Document], save_path: str = "multimodal_kb_index"):
        """构建多模态向量数据库"""
        try:
            logger.info(f"开始构建多模态向量数据库，文档数量: {len(documents)}")
            
            if not documents:
                logger.warning("没有文档需要向量化")
                return
            
            # 使用FAISS构建向量数据库
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.text_embeddings
            )
            
            # 保存向量数据库
            self.vector_db.save_local(save_path)
            
            logger.info(f"✅ 多模态向量数据库构建完成，保存到: {save_path}")
            logger.info(f"向量数据库大小: {len(self.vector_db.index_to_docstore_id)} 个向量")
            
        except Exception as e:
            logger.error(f"❌ 向量数据库构建失败: {e}")
            raise
    
    def search_similar(self, query: str, top_k: int = 5, filter_type: str = None) -> List[Dict[str, Any]]:
        """相似性搜索，支持类型过滤"""
        if not self.vector_db:
            logger.error("向量数据库未初始化")
            return []
        
        try:
            # 执行搜索
            docs_and_scores = self.vector_db.similarity_search_with_score(
                query, 
                k=top_k
            )
            
            results = []
            for doc, score in docs_and_scores:
                # 类型过滤
                if filter_type and doc.metadata.get('type') != filter_type:
                    continue
                
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取向量数据库信息"""
        if not self.vector_db:
            return {"status": "未初始化"}
        
        try:
            # 统计各类型文档数量
            type_counts = {}
            for doc_id in self.vector_db.index_to_docstore_id.values():
                doc = self.vector_db.docstore._dict[doc_id]
                doc_type = doc.metadata.get('type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            return {
                "status": "已初始化",
                "vector_count": len(self.vector_db.index_to_docstore_id),
                "embedding_dimension": self.text_embeddings.get_sentence_embedding_dimension(),
                "text_model": self.text_model,
                "device": self.device,
                "document_types": type_counts
            }
        except Exception as e:
            return {"status": f"获取信息失败: {e}"}

def main():
    """主函数 - 构建多模态企业知识库"""
    print("🚀 多模态企业知识库向量化系统启动")
    print("=" * 60)
    
    try:
        # 1. 初始化多模态向量化器
        vectorizer = MultimodalVectorizer()
        
        # 2. 处理不同类型的文件
        all_documents = []
        
        # 文本文件
        if os.path.exists("enterprise_terminology_complete.json"):
            print("📚 处理术语库...")
            term_docs = vectorizer.process_file("enterprise_terminology_complete.json", {'category': 'terminology'})
            all_documents.extend(term_docs)
            print(f"  - 术语库: {len(term_docs)} 个文档")
        
        # 业务文档
        if os.path.exists("sample_docs"):
            print("📄 处理业务文档...")
            business_docs = vectorizer.process_directory("sample_docs", {'category': 'business'})
            all_documents.extend(business_docs)
            print(f"  - 业务文档: {len(business_docs)} 个文档")
        
        # 图像文件
        if os.path.exists("images"):
            print("🖼️ 处理图像文件...")
            image_docs = vectorizer.process_directory("images", {'category': 'visual'})
            all_documents.extend(image_docs)
            print(f"  - 图像文件: {len(image_docs)} 个文档")
        
        # 视频文件
        if os.path.exists("videos"):
            print("🎥 处理视频文件...")
            video_docs = vectorizer.process_directory("videos", {'category': 'video'})
            all_documents.extend(video_docs)
            print(f"  - 视频文件: {len(video_docs)} 个文档")
        
        # 音频文件
        if os.path.exists("audios"):
            print("🎵 处理音频文件...")
            audio_docs = vectorizer.process_directory("audios", {'category': 'audio'})
            all_documents.extend(audio_docs)
            print(f"  - 音频文件: {len(audio_docs)} 个文档")
        
        print(f"\n📊 总文档数量: {len(all_documents)}")
        
        # 3. 构建向量数据库
        if all_documents:
            vectorizer.build_vector_database(all_documents, "multimodal_kb_index")
            
            # 4. 测试搜索功能
            print("\n🧪 测试多模态搜索功能:")
            test_queries = ["AAP是什么？", "客户经理职责", "打印服务流程", "图表分析", "视频内容"]
            
            for query in test_queries:
                print(f"\n查询: {query}")
                results = vectorizer.search_similar(query, top_k=3)
                
                for i, result in enumerate(results, 1):
                    print(f"  结果 {i}: 相似度 {result['similarity_score']:.4f}")
                    print(f"    类型: {result['metadata'].get('type', 'N/A')}")
                    print(f"    内容: {result['content'][:100]}...")
                    print(f"    来源: {result['metadata'].get('source', 'N/A')}")
            
            # 5. 显示数据库信息
            print("\n📊 多模态向量数据库信息:")
            db_info = vectorizer.get_database_info()
            for key, value in db_info.items():
                print(f"  {key}: {value}")
        
        print("\n🎉 多模态企业知识库向量化完成！")
        print("下一步：集成到千问Agent系统中，支持多模态查询")
        
    except Exception as e:
        print(f"❌ 多模态向量化过程失败: {e}")
        logger.error(f"多模态向量化失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()
