# 🏢 Dify 企业知识库完整配置指南

## 📋 系统架构概览

```
企业文档/视频 → Dify知识库 → 向量化 → 7B LoRA微调 → 多模态模型 → 知识图谱
```

## 🚀 第一步：运行部署脚本

```bash
# 给脚本执行权限
chmod +x dify_enterprise_kb_setup.sh

# 运行部署脚本
./dify_enterprise_kb_setup.sh
```

## ⚙️ 第二步：Dify 基础配置

### 2.1 访问 Dify 安装页面
根据[Dify官方文档](https://docs.dify.ai/zh-hans/getting-started/install-self-hosted/docker-compose)，首次访问需要：

- 浏览器打开：`http://你的服务器IP:3000/install`
- 这是管理员初始化页面，用于设置管理员账户
- **重要**：不要直接访问根路径，必须先访问 `/install` 路径

### 2.2 创建管理员账户
1. 在安装页面填写管理员信息：
   - 邮箱地址
   - 密码
   - 确认密码
2. 点击"创建账户"
3. 等待系统初始化完成

### 2.3 访问 Dify 主界面
- 初始化完成后，访问：`http://你的服务器IP:3000`
- 使用刚创建的管理员账户登录

### 2.4 配置向量数据库
1. 进入 **设置** → **模型提供商**
2. 选择 **向量数据库** → **Milvus**
3. 填写配置：
   ```
   Host: 127.0.0.1
   Port: 19530
   Database: default
   Username: (留空)
   Password: (留空)
   ```

### 2.5 配置 Embedding 模型
1. 在 **模型提供商** 中选择 **Embedding**
2. 选择 **OpenAI-API-Compatible**
3. 填写配置：
   ```
   API Base: http://127.0.0.1:6006/v1
   API Key: (任意填写)
   Model Name: BAAI/bge-large-zh
   ```

## 🎥 第三步：多模态模型配置

### 3.1 视频理解模型
1. 在 **模型提供商** 中选择 **LLM**
2. 选择 **OpenAI-API-Compatible**
3. 填写配置：
   ```
   API Base: http://127.0.0.1:9000/v1
   API Key: (任意填写)
   Model Name: minicpm-video
   ```

### 3.2 图像理解模型
1. 再次添加 **LLM** 提供商
2. 选择 **OpenAI-API-Compatible**
3. 填写配置：
   ```
   API Base: http://127.0.0.1:9001/v1
   API Key: (任意填写)
   Model Name: qwen-vl
   ```

## 🤖 第四步：7B LoRA 微调配置

### 4.1 准备训练数据
根据你的企业文档，创建训练数据：

```bash
cd ~/LLaMA-Factory
nano data/enterprise_kb.jsonl
```

数据格式示例：
```json
{"instruction":"你的问题","input":"","output":"标准答案"}
{"instruction":"ESS培训每月几次？","input":"","output":"每月3次，详见《ESS培训管理制度》"}
{"instruction":"什么是ESS系统？","input":"","output":"ESS是员工自助服务系统，用于请假申请、考勤查询等"}
```

### 4.2 执行 LoRA 微调
```bash
# 激活虚拟环境
source venv/bin/activate

# 开始微调（约30-60分钟）
python src/train_bash.py \
  --stage sft \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset enterprise_kb \
  --template qwen \
  --finetuning_type lora \
  --output_dir ./lora_ckpt \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --quantization_bit 4 \
  --learning_rate 3e-4 \
  --fp16
```

### 4.3 启动 LoRA 推理服务
```bash
# 使用 vLLM 启动带 LoRA 的模型
docker run -d --name vllm_lora \
  -p 8000:8000 \
  -v $(pwd)/lora_ckpt:/lora \
  vllm/vllm-openai:latest \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --enable-lora \
  --lora-modules enterprise_lora=/lora
```

### 4.4 在 Dify 中配置 LoRA 模型
1. 添加新的 **LLM** 提供商
2. 选择 **OpenAI-API-Compatible**
3. 填写配置：
   ```
   API Base: http://127.0.0.1:8000/v1
   API Key: (任意填写)
   Model Name: enterprise_lora
   ```

## 📚 第五步：创建知识库

### 5.1 上传企业文档
1. 进入 **知识库** → **新建知识库**
2. 填写基本信息：
   - 名称：企业知识库
   - 描述：包含公司制度、流程、简称等
3. 选择 **Embedding 模型**：BAAI/bge-large-zh
4. 选择 **向量数据库**：Milvus

### 5.2 上传文档
1. 点击 **上传文件**
2. 上传你的企业文档：
   - Q&A 文档
   - 内部简称文档
   - 其他相关文档
3. 等待文档解析和向量化完成

### 5.3 配置文档处理规则
1. **文本分割**：选择 **智能分割**
2. **元数据提取**：启用
3. **表格识别**：启用
4. **图片 OCR**：启用

## 🧠 第六步：配置知识图谱

### 6.1 创建知识图谱应用
1. 进入 **应用** → **新建应用**
2. 选择 **对话机器人**
3. 配置基本信息

### 6.2 设置对话流程
1. **系统提示词**：
```
你是一个专业的企业知识库助手，专门回答关于公司制度、流程、简称等问题。
请基于知识库内容提供准确、详细的答案，并注明信息来源。
对于内部简称，请提供完整的中文名称和简要说明。
```

2. **对话模型**：选择你的 LoRA 微调模型
3. **知识库**：选择刚创建的企业知识库

### 6.3 配置知识图谱规则
在 **提示词编排** 中添加：

```
# 知识图谱查询规则
如果用户询问：
- 公司内部简称 → 查询简称文档，提供完整名称和说明
- 具体流程 → 查询相关制度文档，提供步骤和注意事项
- 培训安排 → 查询相关文档，提供时间和要求

# 多模态处理
如果文档包含：
- 表格 → 提取表格数据，结构化展示
- 图片 → 使用图像理解模型分析内容
- 视频 → 使用视频理解模型提取关键信息
```

## 🎯 第七步：测试和优化

### 7.1 基础功能测试
测试以下问题：
- "ESS培训每月几次？"
- "什么是ESS系统？"
- "公司内部简称有哪些？"
- "如何申请年假？"

### 7.2 多模态测试
- 上传包含表格的文档，测试表格解析
- 上传包含图片的文档，测试图像理解
- 上传视频文件，测试视频内容提取

### 7.3 性能优化
1. **向量检索优化**：
   - 调整相似度阈值
   - 优化检索数量（top-k）
   
2. **模型响应优化**：
   - 调整温度参数
   - 优化最大token数

## 🔧 常见问题解决

### Q1: 无法访问 Dify 安装页面
```bash
# 检查 Dify 服务状态
docker ps | grep dify

# 检查端口是否被占用
netstat -tlnp | grep :3000

# 查看 Dify 日志
docker logs dify-web-1
```

### Q2: 向量数据库连接失败
```bash
# 检查 Milvus 状态
docker ps | grep milvus

# 重启 Milvus
docker restart milvus
```

### Q3: Embedding 服务无响应
```bash
# 检查服务状态
curl http://localhost:6006/health

# 重启服务
docker restart bge_embedding
```

### Q4: LoRA 模型加载失败
```bash
# 检查模型文件
ls -la ~/LLaMA-Factory/lora_ckpt/

# 重新启动 vLLM 服务
docker restart vllm_lora
```

## 📊 监控和维护

### 系统监控
```bash
# 查看所有服务状态
docker ps

# 查看 Dify 服务状态
cd ~/dify/docker
docker compose ps

# 查看服务日志
docker logs dify-web-1
docker logs dify-api-1
docker logs milvus
docker logs bge_embedding
```

### 定期维护
1. **数据更新**：定期上传新的企业文档
2. **模型优化**：根据用户反馈调整 LoRA 训练数据
3. **性能监控**：关注响应时间和准确率

## 🎉 完成！

现在你拥有了一个完整的企业知识库系统，包含：
- ✅ Dify 管理平台 (基于官方 0.15.3 版本)
- ✅ 多模态文档处理
- ✅ 7B LoRA 微调模型
- ✅ 知识图谱和智能问答
- ✅ 向量数据库存储

可以开始上传你的企业文档和视频，测试系统功能了！

## 📚 参考文档

- [Dify 官方部署文档](https://docs.dify.ai/zh-hans/getting-started/install-self-hosted/docker-compose)
- [Dify GitHub 仓库](https://github.com/langgenius/dify)
