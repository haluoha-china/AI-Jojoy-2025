#!/bin/bash

echo "🔧 安装多模态系统依赖 (修复版本兼容性)"
echo "=================================="

# 设置pip镜像源
export PIP_INDEX_URL="http://mirrors.aliyun.com/pypi/simple"

# 1. 基础图像处理库
echo "📸 安装图像处理库..."
pip install pillow==9.5.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install plotly==5.15.0

# 2. OCR和图表理解
echo "🔍 安装OCR和图表理解库..."
pip install easyocr==1.7.0
pip install paddlepaddle-gpu==2.5.2
pip install paddleocr==2.7.0.3

# 3. Transformers (使用兼容版本)
echo "🤖 安装Transformers (兼容版本)..."
pip install transformers>=4.40.0
pip install torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.9.12

# 4. 视频处理
echo "🎥 安装视频处理库..."
pip install opencv-python-headless==4.8.0.76
pip install moviepy==1.0.3
pip install imageio==2.31.1

# 5. 音频处理
echo "🎵 安装音频处理库..."
pip install librosa==0.10.0
pip install soundfile==0.12.1
pip install pyav==10.0.0

# 6. 科学计算库 (使用兼容版本)
echo "🧮 安装科学计算库..."
pip install scipy==1.10.1
pip install scikit-learn==1.3.0

# 7. 工具库 (使用兼容版本)
echo "🛠️ 安装工具库..."
pip install tqdm==4.66.3
pip install colorama==0.4.6
pip install rich==13.5.2

# 8. 验证安装
echo "✅ 验证安装结果..."
python -c "
import sys
print('='*60)
print('Python版本:', sys.version)
print('='*60)

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ OpenCV:', e)

try:
    import easyocr
    print('✅ EasyOCR:', easyocr.__version__)
except ImportError as e:
    print('❌ EasyOCR:', e)

try:
    import transformers
    print('✅ Transformers:', transformers.__version__)
except ImportError as e:
    print('❌ Transformers:', e)

try:
    import librosa
    print('✅ Librosa:', librosa.__version__)
except ImportError as e:
    print('❌ Librosa:', e)

try:
    import scipy
    print('✅ SciPy:', scipy.__version__)
except ImportError as e:
    print('❌ SciPy:', e)

try:
    import sklearn
    print('✅ Scikit-learn:', sklearn.__version__)
except ImportError as e:
    print('❌ Scikit-learn:', e)

print('='*60)
print('🎉 多模态依赖安装完成！')
print('='*60)
"

echo "🚀 现在可以运行多模态系统了！"
