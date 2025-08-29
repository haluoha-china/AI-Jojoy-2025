#!/bin/bash
"""
多模态向量化系统依赖安装脚本
安装OCR、图表理解、视频和音频处理所需的库
"""

echo "🚀 开始安装多模态向量化系统依赖..."
echo "=" * 60

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️ 警告：未检测到conda环境，请先激活kb_enterprise环境"
    echo "命令: conda activate kb_enterprise"
    exit 1
fi

echo "✅ 当前conda环境: $CONDA_DEFAULT_ENV"

# 1. 安装OCR相关依赖
echo "📖 安装OCR相关依赖..."
pip install easyocr==1.7.0
pip install paddlepaddle-gpu==2.5.2
pip install paddleocr==2.7.0.3

# 2. 安装图表理解依赖
echo "📊 安装图表理解依赖..."
pip install transformers==4.38.2
pip install torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.9.12

# 3. 安装图像处理增强库
echo "🖼️ 安装图像处理增强库..."
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install plotly==5.17.0
pip install imageio==2.31.1
pip install scikit-image==0.21.0

# 4. 安装音频处理增强库
echo "🎵 安装音频处理增强库..."
pip install soundfile==0.12.1
pip install pydub==0.25.1
pip install webrtcvad==2.0.10

# 5. 安装视频处理增强库
echo "🎥 安装视频处理增强库..."
pip install imageio-ffmpeg==0.4.9
pip install av==10.0.0

# 6. 安装机器学习工具
echo "🤖 安装机器学习工具..."
pip install scikit-learn==1.3.0
pip install scipy==1.11.3

# 7. 安装其他工具库
echo "🔧 安装其他工具库..."
pip install tqdm==4.66.1
pip install colorama==0.4.6
pip install rich==13.6.0

echo "✅ 所有依赖安装完成！"
echo "=" * 60

# 验证安装
echo "🔍 验证关键依赖安装..."
python -c "
try:
    import easyocr
    print('✅ EasyOCR:', easyocr.__version__)
except ImportError:
    print('❌ EasyOCR 未安装')

try:
    import paddlepaddle
    print('✅ PaddlePaddle:', paddlepaddle.__version__)
except ImportError:
    print('❌ PaddlePaddle 未安装')

try:
    import transformers
    print('✅ Transformers:', transformers.__version__)
except ImportError:
    print('❌ Transformers 未安装')

try:
    import matplotlib
    print('✅ Matplotlib:', matplotlib.__version__)
except ImportError:
    print('❌ Matplotlib 未安装')

try:
    import seaborn
    print('✅ Seaborn:', seaborn.__version__)
except ImportError:
    print('❌ Seaborn 未安装')

try:
    import plotly
    print('✅ Plotly:', plotly.__version__)
except ImportError:
    print('❌ Plotly 未安装')

try:
    import soundfile
    print('✅ SoundFile:', soundfile.__version__)
except ImportError:
    print('❌ SoundFile 未安装')

try:
    import av
    print('✅ PyAV:', av.__version__)
except ImportError:
    print('❌ PyAV 未安装')
"

echo ""
echo "🎉 依赖安装完成！现在可以运行测试脚本："
echo "python test_multimodal_system.py"
echo ""
echo "或者直接运行多模态向量化系统："
echo "python multimodal_vectorizer.py"
