# 🚀 快速启动检查清单

## ✅ 部署前检查

- [ ] 确认服务器有 RTX 4090 24GB GPU
- [ ] 确认 Ubuntu 22.04 系统
- [ ] 确认网络连接正常
- [ ] 确认有足够的磁盘空间（建议 100GB+）

## 🚀 一键部署

```bash
# 1. 给脚本执行权限
chmod +x dify_enterprise_kb_setup.sh

# 2. 运行部署脚本
./dify_enterprise_kb_setup.sh
```

## 🔍 部署后验证

### 1. 检查服务状态
```bash
# 查看所有容器状态
docker ps

# 应该看到以下服务：
# - milvus (向量数据库)
# - bge_embedding (中文Embedding)
# - minicpm_video (视频理解)
# - qwen_vl (图像理解)
# - dify-web (Dify平台)
```

### 2. 检查端口开放
```bash
# 检查关键端口
netstat -tlnp | grep -E ':(3000|19530|6006|9000|9001|8000)'

# 应该看到：
# 3000  - Dify Web界面
# 19530 - Milvus向量库
# 6006  - Embedding服务
# 9000  - 视频理解API
# 9001  - 图像理解API
# 8000  - LoRA推理服务
```

### 3. 测试服务连通性
```bash
# 测试 Milvus
curl http://localhost:19530/health

# 测试 Embedding 服务
curl http://localhost:6006/health

# 测试视频理解服务
curl http://localhost:9000/health

# 测试图像理解服务
curl http://localhost:9001/health
```

## 🌐 访问验证

### 1. Dify 管理界面
- 浏览器打开：`http://你的服务器IP:3000`
- 创建管理员账户
- 进入设置页面

### 2. 配置模型提供商
- [ ] 配置 Milvus 向量数据库
- [ ] 配置 BGE Embedding 模型
- [ ] 配置多模态模型
- [ ] 配置 LoRA 模型

### 3. 创建知识库
- [ ] 新建知识库
- [ ] 上传企业文档
- [ ] 等待向量化完成

## 🧪 功能测试

### 1. 基础问答测试
```bash
# 测试简称查询
"什么是ESS？"

# 测试流程查询
"如何申请年假？"

# 测试制度查询
"公司培训制度是什么？"
```

### 2. 多模态测试
- [ ] 上传包含表格的文档
- [ ] 上传包含图片的文档
- [ ] 上传视频文件
- [ ] 测试各种格式的解析

### 3. LoRA 微调测试
```bash
# 检查训练环境
cd ~/LLaMA-Factory
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 开始微调
python src/train_bash.py --stage sft --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dataset enterprise_kb --template qwen --finetuning_type lora --output_dir ./lora_ckpt --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --num_train_epochs 3 --quantization_bit 4 --learning_rate 3e-4 --fp16
```

## 🔧 常见问题快速解决

### 问题1：Docker 服务启动失败
```bash
# 重启 Docker 服务
sudo systemctl restart docker

# 检查 Docker 状态
sudo systemctl status docker
```

### 问题2：端口被占用
```bash
# 查看端口占用
sudo lsof -i :3000

# 杀死占用进程
sudo kill -9 <PID>
```

### 问题3：GPU 内存不足
```bash
# 检查 GPU 状态
nvidia-smi

# 清理 GPU 缓存
sudo fuser -v /dev/nvidia*
```

### 问题4：模型下载失败
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 📊 性能监控

### 1. 系统资源监控
```bash
# 查看系统资源
htop

# 查看 GPU 使用情况
watch -n 1 nvidia-smi

# 查看磁盘使用情况
df -h
```

### 2. 服务性能监控
```bash
# 查看容器资源使用
docker stats

# 查看服务日志
docker logs -f dify-web
docker logs -f milvus
```

## 🎯 成功标志

当看到以下情况时，说明系统部署成功：

✅ **服务状态**：所有 Docker 容器都显示 "Up" 状态
✅ **端口开放**：所有关键端口都能正常访问
✅ **Web界面**：Dify 管理界面能正常打开和登录
✅ **模型配置**：所有模型提供商配置成功
✅ **知识库**：能正常上传文档并完成向量化
✅ **问答功能**：能正常回答企业相关问题
✅ **多模态**：能处理文档、图片、视频等不同格式

## 🚨 紧急联系

如果遇到无法解决的问题：

1. **检查日志**：`docker logs <容器名>`
2. **重启服务**：`docker restart <容器名>`
3. **查看状态**：`docker ps` 和 `docker stats`
4. **系统重启**：`sudo reboot`（最后手段）

## 🎉 恭喜！

如果所有检查项都通过，恭喜你成功部署了企业知识库系统！

现在可以：
- 上传你的企业文档和视频
- 配置 LoRA 微调
- 测试各种功能
- 开始实际使用

祝你使用愉快！🚀
