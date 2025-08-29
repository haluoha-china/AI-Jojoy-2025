#!/usr/bin/env python3
"""
多模态向量化系统测试脚本
测试OCR、图表理解、视频和音频处理功能
"""

import os
import sys
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """测试环境依赖"""
    print("🔍 测试环境依赖...")
    
    try:
        # 测试基础库
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        from PIL import Image
        print("✅ PIL/Pillow: 可用")
        
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        
        import seaborn
        print(f"✅ Seaborn: {seaborn.__version__}")
        
        import plotly
        print(f"✅ Plotly: {plotly.__version__}")
        
        # 测试音频处理
        import librosa
        print(f"✅ Librosa: {librosa.__version__}")
        
        # 测试文本处理
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers: 可用")
        
        from langchain_community.vectorstores import FAISS
        print("✅ LangChain FAISS: 可用")
        
        # 测试OCR
        import easyocr
        print("✅ EasyOCR: 可用")
        
        # 测试图表理解
        from transformers import pipeline
        print("✅ Transformers Pipeline: 可用")
        
        print("\n🎉 所有环境依赖检查通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 环境依赖缺失: {e}")
        return False
    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        return False

def test_ocr_functionality():
    """测试OCR功能"""
    print("\n🔍 测试OCR功能...")
    
    try:
        import easyocr
        import cv2
        import numpy as np
        
        # 创建一个测试图像
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # 在图像上添加文字
        cv2.putText(test_image, "Hello World", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "测试文字", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 保存测试图像
        test_image_path = "test_ocr_image.jpg"
        cv2.imwrite(test_image_path, test_image)
        
        # 初始化OCR
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # 使用CPU避免GPU内存问题
        
        # 执行OCR
        results = reader.readtext(test_image_path)
        
        print(f"✅ OCR识别结果: {len(results)} 个文本区域")
        for i, (bbox, text, confidence) in enumerate(results):
            print(f"  文本 {i+1}: '{text}' (置信度: {confidence:.3f})")
        
        # 清理测试文件
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ OCR测试失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n🔍 测试图像处理功能...")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # 保存测试图像
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, test_image)
        
        # 读取图像
        image = cv2.imread(test_image_path)
        
        # 基本图像信息
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算统计信息
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        print(f"✅ 图像信息: {width}x{height}, 通道数: {channels}")
        print(f"✅ 亮度: {brightness:.1f}, 对比度: {contrast:.1f}")
        
        # 测试PIL
        pil_image = Image.open(test_image_path)
        print(f"✅ PIL图像尺寸: {pil_image.size}")
        
        # 清理测试文件
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 图像处理测试失败: {e}")
        return False

def test_audio_processing():
    """测试音频处理功能"""
    print("\n🔍 测试音频处理功能...")
    
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        
        # 创建测试音频（1秒的440Hz正弦波）
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # 保存测试音频
        test_audio_path = "test_audio.wav"
        sf.write(test_audio_path, test_audio, sample_rate)
        
        # 加载音频
        y, sr = librosa.load(test_audio_path, sr=None)
        
        # 音频特征
        duration_actual = librosa.get_duration(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        print(f"✅ 音频信息: 时长 {duration_actual:.2f}秒, 采样率 {sr}Hz")
        print(f"✅ 节拍: {tempo:.1f} BPM, 检测到 {len(beats)} 个节拍")
        print(f"✅ 频谱质心: {np.mean(spectral_centroids):.1f} Hz")
        
        # 清理测试文件
        os.remove(test_audio_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 音频处理测试失败: {e}")
        return False

def test_video_processing():
    """测试视频处理功能"""
    print("\n🔍 测试视频处理功能...")
    
    try:
        import cv2
        import numpy as np
        
        # 创建测试视频
        test_video_path = "test_video.avi"
        
        # 视频参数
        fps = 10
        duration = 2  # 2秒
        width, height = 320, 240
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
        
        # 生成测试帧
        for i in range(fps * duration):
            # 创建彩色帧
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # 添加帧号
            cv2.putText(frame, f"Frame {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        # 读取视频信息
        cap = cv2.VideoCapture(test_video_path)
        
        if cap.isOpened():
            fps_actual = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ 视频信息: {width_actual}x{height_actual}, {fps_actual:.1f}fps")
            print(f"✅ 总帧数: {frame_count}, 时长: {frame_count/fps_actual:.1f}秒")
            
            # 读取第一帧
            ret, frame = cap.read()
            if ret:
                print(f"✅ 成功读取第一帧: {frame.shape}")
            
            cap.release()
        
        # 清理测试文件
        os.remove(test_video_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 视频处理测试失败: {e}")
        return False

def test_text_embeddings():
    """测试文本嵌入功能"""
    print("\n🔍 测试文本嵌入功能...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 初始化模型（使用较小的模型进行测试）
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 较小的模型
        
        # 测试文本
        texts = ['Hello World', '你好世界', '这是一个测试']
        
        # 生成嵌入
        embeddings = model.encode(texts)
        
        print(f"✅ 嵌入维度: {embeddings.shape}")
        print(f"✅ 第一个向量: {embeddings[0][:5]}...")
        
        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        print("✅ 文本相似度矩阵:")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                print(f"  '{text1}' vs '{text2}': {similarity_matrix[i][j]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本嵌入测试失败: {e}")
        return False

def test_faiss_integration():
    """测试FAISS集成"""
    print("\n🔍 测试FAISS集成...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_community.vectorstores import FAISS
        
        # 初始化模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 创建测试文档
        from langchain.schema import Document
        
        test_docs = [
            Document(page_content="这是第一个测试文档", metadata={'id': 1}),
            Document(page_content="这是第二个测试文档", metadata={'id': 2}),
            Document(page_content="这是第三个测试文档", metadata={'id': 3})
        ]
        
        # 创建向量数据库
        vector_db = FAISS.from_documents(
            documents=test_docs,
            embedding=model
        )
        
        print(f"✅ FAISS数据库创建成功，包含 {len(vector_db.index_to_docstore_id)} 个向量")
        
        # 测试搜索
        query = "测试文档"
        results = vector_db.similarity_search_with_score(query, k=2)
        
        print(f"✅ 搜索查询 '{query}' 返回 {len(results)} 个结果:")
        for i, (doc, score) in enumerate(results):
            print(f"  结果 {i+1}: 相似度 {score:.4f}, 内容: {doc.page_content}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS集成测试失败: {e}")
        return False

def create_sample_data():
    """创建示例数据目录和文件"""
    print("\n📁 创建示例数据...")
    
    try:
        # 创建目录
        directories = ['sample_docs', 'images', 'videos', 'audios']
        for dir_name in directories:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ 创建目录: {dir_name}")
        
        # 创建示例文本文件
        sample_text = """企业知识库示例文档

这是一个示例业务文档，用于测试多模态向量化系统。

主要内容包括：
1. 企业术语解释
2. 业务流程说明
3. 技术规范文档

这个文档将被系统自动分割和向量化，用于后续的语义搜索。"""
        
        with open('sample_docs/sample_business_doc.txt', 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print("✅ 创建示例业务文档")
        
        # 创建示例图像描述文件
        image_desc = """示例图像描述

这是一个包含图表的业务报告图像，显示了：
- 销售数据趋势
- 市场份额分析
- 客户满意度统计

图像中的文字将通过OCR进行识别和提取。"""
        
        with open('images/image_description.txt', 'w', encoding='utf-8') as f:
            f.write(image_desc)
        print("✅ 创建图像描述文件")
        
        # 创建示例视频描述文件
        video_desc = """示例视频描述

这是一个产品演示视频，内容包括：
- 产品功能介绍
- 操作流程演示
- 常见问题解答

视频将通过关键帧提取和音频分析进行处理。"""
        
        with open('videos/video_description.txt', 'w', encoding='utf-8') as f:
            f.write(video_desc)
        print("✅ 创建视频描述文件")
        
        # 创建示例音频描述文件
        audio_desc = """示例音频描述

这是一个客户服务录音，内容包括：
- 客户咨询问题
- 服务人员解答
- 问题解决过程

音频将通过频谱分析和特征提取进行处理。"""
        
        with open('audios/audio_description.txt', 'w', encoding='utf-8') as f:
            f.write(audio_desc)
        print("✅ 创建音频描述文件")
        
        print("✅ 示例数据创建完成！")
        return True
        
    except Exception as e:
        print(f"❌ 示例数据创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 多模态向量化系统功能测试")
    print("=" * 60)
    
    test_results = []
    
    # 1. 环境依赖测试
    test_results.append(("环境依赖", test_environment()))
    
    # 2. 各功能模块测试
    test_results.append(("OCR功能", test_ocr_functionality()))
    test_results.append(("图像处理", test_image_processing()))
    test_results.append(("音频处理", test_audio_processing()))
    test_results.append(("视频处理", test_video_processing()))
    test_results.append(("文本嵌入", test_text_embeddings()))
    test_results.append(("FAISS集成", test_faiss_integration()))
    
    # 3. 创建示例数据
    test_results.append(("示例数据", create_sample_data()))
    
    # 4. 显示测试结果
    print("\n📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"总测试数: {total}, 通过: {passed}, 失败: {total - passed}")
    
    if passed == total:
        print("\n🎉 所有测试通过！多模态向量化系统准备就绪！")
        print("\n下一步：运行完整的多模态向量化流程")
        print("命令: python multimodal_vectorizer.py")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查相关依赖")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
