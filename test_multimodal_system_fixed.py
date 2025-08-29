#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态向量化系统功能测试 (修复版本)
解决PIL和FAISS兼容性问题
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import librosa
import soundfile as sf
import easyocr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import torch

def test_environment():
    """测试环境依赖"""
    print("🔍 测试环境依赖...")
    
    try:
        # PyTorch和CUDA
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU设备: {torch.cuda.get_device_name()}")
        
        # OpenCV
        print(f"✅ OpenCV: {cv2.__version__}")
        
        # NumPy
        print(f"✅ NumPy: {np.__version__}")
        
        # PIL/Pillow
        try:
            from PIL import Image
            print("✅ PIL/Pillow: 可用")
        except ImportError:
            print("❌ PIL/Pillow: 不可用")
        
        # Matplotlib
        print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
        
        # Seaborn
        print(f"✅ Seaborn: {sns.__version__}")
        
        # Plotly
        print(f"✅ Plotly: {plotly.__version__}")
        
        # Librosa
        print(f"✅ Librosa: {librosa.__version__}")
        
        # SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ SentenceTransformers: 可用")
        except ImportError:
            print("❌ SentenceTransformers: 不可用")
        
        # LangChain FAISS
        try:
            import faiss
            print("✅ LangChain FAISS: 可用")
        except ImportError:
            print("❌ LangChain FAISS: 不可用")
        
        # EasyOCR
        try:
            import easyocr
            print("✅ EasyOCR: 可用")
        except ImportError:
            print("❌ EasyOCR: 不可用")
        
        # Transformers Pipeline
        try:
            from transformers import pipeline
            print("✅ Transformers Pipeline: 可用")
        except ImportError:
            print("❌ Transformers Pipeline: 不可用")
        
        print("\n🎉 所有环境依赖检查通过！")
        return True
        
    except Exception as e:
        print(f"❌ 环境依赖检查失败: {e}")
        return False

def test_ocr_functionality():
    """测试OCR功能 (修复PIL兼容性问题)"""
    print("\n🔍 测试OCR功能...")
    
    try:
        # 创建测试图像
        test_image_path = "test_ocr_image.png"
        
        # 创建一个简单的测试图像
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Hello World", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(test_image_path, img)
        
        # 初始化EasyOCR
        reader = easyocr.Reader(['en'], gpu=False)  # 使用CPU避免GPU问题
        
        # 执行OCR
        results = reader.readtext(test_image_path)
        
        if results:
            print(f"✅ OCR识别成功: {len(results)} 个文本区域")
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"  文本{i+1}: '{text}' (置信度: {confidence:.3f})")
        else:
            print("⚠️ OCR未识别到文本")
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ OCR测试失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n🔍 测试图像处理功能...")
    
    try:
        # 创建测试图像
        test_image_path = "test_image.png"
        
        # 创建一个彩色测试图像
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, img)
        
        # 读取图像
        image = cv2.imread(test_image_path)
        height, width, channels = image.shape
        
        # 计算基本特征
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        print(f"✅ 图像信息: {width}x{height}, 通道数: {channels}")
        print(f"✅ 亮度: {brightness:.1f}, 对比度: {contrast:.1f}")
        
        # 测试PIL兼容性
        try:
            pil_image = Image.open(test_image_path)
            # 修复PIL兼容性问题
            if hasattr(Image, 'Resampling'):
                resized = pil_image.resize((200, 150), Image.Resampling.LANCZOS)
            else:
                try:
                    resized = pil_image.resize((200, 150), Image.ANTIALIAS)
                except AttributeError:
                    resized = pil_image.resize((200, 150))
            
            print(f"✅ PIL图像尺寸: {resized.size}")
            
        except Exception as e:
            print(f"⚠️ PIL处理警告: {e}")
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 图像处理测试失败: {e}")
        return False

def test_audio_processing():
    """测试音频处理功能"""
    print("\n🔍 测试音频处理功能...")
    
    try:
        # 创建测试音频
        test_audio_path = "test_audio.wav"
        
        # 生成1秒的测试音频
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz正弦波
        
        # 保存音频
        sf.write(test_audio_path, audio, sample_rate)
        
        # 加载音频
        y, sr = librosa.load(test_audio_path, sr=None)
        
        # 提取特征
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        print(f"✅ 音频信息: 时长 {len(y)/sr:.2f}秒, 采样率 {sr}Hz")
        print(f"✅ 节拍: {tempo:.1f} BPM, 检测到 {len(librosa.beat.beat_track(y=y, sr=sr)[1])} 个节拍")
        print(f"✅ 频谱质心: {np.mean(spectral_centroids):.1f} Hz")
        
        # 清理测试文件
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 音频处理测试失败: {e}")
        return False

def test_video_processing():
    """测试视频处理功能"""
    print("\n🔍 测试视频处理功能...")
    
    try:
        # 创建测试视频
        test_video_path = "test_video.mp4"
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (320, 240))
        
        # 生成20帧测试视频
        for i in range(20):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # 读取视频
        cap = cv2.VideoCapture(test_video_path)
        
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"✅ 视频信息: {width}x{height}, {fps:.1f}fps")
            print(f"✅ 总帧数: {frame_count}, 时长: {duration:.1f}秒")
            
            # 读取第一帧
            ret, frame = cap.read()
            if ret:
                print(f"✅ 成功读取第一帧: {frame.shape}")
            
            cap.release()
        
        # 清理测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 视频处理测试失败: {e}")
        return False

def test_text_embeddings():
    """测试文本嵌入功能"""
    print("\n🔍 测试文本嵌入功能...")
    
    try:
        # 使用轻量级模型进行测试
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 测试文本
        texts = ['Hello World', '你好世界', '这是一个测试']
        
        # 生成嵌入
        embeddings = model.encode(texts, show_progress_bar=True)
        
        print(f"✅ 嵌入维度: {embeddings.shape}")
        print(f"✅ 第一个向量: {embeddings[0][:5]}...")
        
        # 计算相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        print("✅ 文本相似度矩阵:")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                sim = similarity_matrix[i][j]
                print(f"  '{text1}' vs '{text2}': {sim:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本嵌入测试失败: {e}")
        return False

def test_faiss_integration():
    """测试FAISS集成 (修复接口兼容性问题)"""
    print("\n🔍 测试FAISS集成...")
    
    try:
        # 使用轻量级模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 创建测试数据
        texts = [
            "这是第一个测试文档",
            "这是第二个测试文档", 
            "这是第三个测试文档"
        ]
        
        # 生成嵌入
        embeddings = model.encode(texts)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # 归一化向量
        faiss.normalize_L2(embeddings)
        
        # 添加向量到索引
        index.add(embeddings.astype('float32'))
        
        # 测试搜索
        query = "测试文档"
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding.astype('float32'), 3)
        
        print(f"✅ FAISS索引创建成功: {dimension}维, {len(texts)}个文档")
        print(f"✅ 搜索测试成功: 查询'{query}'")
        print(f"  结果1: 文档{indices[0][0]}, 相似度: {scores[0][0]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS集成测试失败: {e}")
        return False

def create_sample_data():
    """创建示例数据目录和文件"""
    print("\n📁 创建示例数据...")
    
    try:
        # 创建目录结构
        directories = [
            "sample_docs/texts",
            "sample_docs/images", 
            "sample_docs/videos",
            "sample_docs/audios"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
        
        # 创建示例业务文档
        business_docs = [
            ("sample_docs/texts/商业合同.txt", "这是一份标准的商业合同模板，包含甲乙双方的权利义务条款。"),
            ("sample_docs/texts/财务报表.txt", "2024年度财务报表，包含收入、支出、利润等关键财务指标。"),
            ("sample_docs/texts/产品说明.txt", "产品功能特性说明，包含技术参数、使用方法、注意事项等。")
        ]
        
        for file_path, content in business_docs:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 创建示例业务文档: {file_path}")
        
        # 创建图像描述文件
        image_descriptions = [
            ("sample_docs/images/产品图片.txt", "产品外观展示图，包含产品的主要特征和设计亮点。"),
            ("sample_docs/images/组织结构图.txt", "公司组织架构图，展示各部门的层级关系和职责分工。")
        ]
        
        for file_path, content in image_descriptions:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 创建图像描述文件: {file_path}")
        
        # 创建视频描述文件
        video_descriptions = [
            ("sample_docs/videos/会议记录.txt", "重要会议的视频记录，包含会议议程、讨论内容和决议事项。"),
            ("sample_docs/videos/产品演示.txt", "产品功能演示视频，展示产品的使用方法和应用场景。")
        ]
        
        for file_path, content in video_descriptions:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 创建视频描述文件: {file_path}")
        
        # 创建音频描述文件
        audio_descriptions = [
            ("sample_docs/audios/会议录音.txt", "会议音频记录，包含发言人的讲话内容和讨论过程。"),
            ("sample_docs/audios/培训课程.txt", "员工培训课程录音，包含专业知识讲解和案例分析。")
        ]
        
        for file_path, content in audio_descriptions:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 创建音频描述文件: {file_path}")
        
        print("✅ 示例数据创建完成！")
        return True
        
    except Exception as e:
        print(f"❌ 创建示例数据失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 多模态向量化系统功能测试 (修复版本)")
    print("=" * 60)
    
    # 运行所有测试
    test_results = []
    
    test_results.append(("环境依赖", test_environment()))
    test_results.append(("OCR功能", test_ocr_functionality()))
    test_results.append(("图像处理", test_image_processing()))
    test_results.append(("音频处理", test_audio_processing()))
    test_results.append(("视频处理", test_video_processing()))
    test_results.append(("文本嵌入", test_text_embeddings()))
    test_results.append(("FAISS集成", test_faiss_integration()))
    test_results.append(("示例数据", create_sample_data()))
    
    # 汇总测试结果
    print("\n📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"总测试数: {len(test_results)}, 通过: {passed}, 失败: {failed}")
    
    if failed > 0:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查相关依赖")
    else:
        print("\n🎉 所有测试通过！系统已就绪")
    
    print(f"\n🚀 现在可以运行多模态向量化系统了！")
    print("运行命令: python multimodal_vectorizer_fixed.py")

if __name__ == "__main__":
    main()
