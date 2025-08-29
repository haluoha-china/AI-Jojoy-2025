# 🚀 autodl环境快速修复指南

## 🔍 **你的autodl环境现状**

根据你提供的信息，你在autodl上的情况是：

- **项目路径**: `/root/autodl-tmp/enterprise_kb/LLaMA-Factory`
- **现有文件**: `data/comprehensive_eval.jsonl` (270行数据)
- **问题**: 缺少训练集文件和正确的配置

## ✅ **立即修复步骤**

### 步骤1: 上传修复脚本到autodl

将 `fix_autodl_training.py` 上传到你的autodl环境，或者直接在autodl上创建这个文件。

### 步骤2: 在autodl上运行修复脚本

```bash
# 进入项目目录
cd /root/autodl-tmp/enterprise_kb/LLaMA-Factory

# 运行修复脚本
python fix_autodl_training.py
```

### 步骤3: 检查生成的文件

```bash
# 检查生成的文件
ls -la train_data_final.jsonl eval_data_final.jsonl
ls -la data/dataset_info.json
ls -la correct_training_command.sh

# 查看数据统计
wc -l train_data_final.jsonl eval_data_final.jsonl
```

### 步骤4: 清理输出目录

```bash
# 清理之前的训练输出
rm -rf ./lora_ckpt_prod
```

### 步骤5: 开始训练

```bash
# 方法1: 直接运行生成的脚本
bash correct_training_command.sh

# 方法2: 手动运行训练命令
python src/train.py \
  --model_name_or_path /root/autodl-tmp/enterprise_kb/models/transformers/DeepSeek-R1-Distill-Qwen-7B \
  --dataset company_abbreviations_train \
  --eval_dataset company_abbreviations_eval \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target q_proj,v_proj \
  --output_dir ./lora_ckpt_prod \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --cutoff_len 256 \
  --save_strategy steps \
  --save_steps 100 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --bf16 true \
  --metric_for_best_model eval_loss \
  --load_best_model_at_end true
```

## 🔑 **关键修复点说明**

### ✅ **数据集名称正确性**
- **`--dataset company_abbreviations_train`**: 对应 `dataset_info.json` 中的键名
- **`--eval_dataset company_abbreviations_eval`**: 对应 `dataset_info.json` 中的键名

### ✅ **文件映射关系**
```json
{
  "company_abbreviations_train": {
    "file_name": "train_data_final.jsonl",  // 实际文件名
    "columns": {
      "prompt": "instruction",
      "query": "input", 
      "response": "output"
    }
  }
}
```

### ✅ **数据分割策略**
- 将现有的270条数据按缩略语分组
- 每个缩略语的80%样本用于训练，20%用于验证
- 确保每个缩略语都有训练和验证样本

## 📊 **预期结果**

修复完成后，你应该看到：

```
=== autodl环境检查 ===
当前目录: /root/autodl-tmp/enterprise_kb/LLaMA-Factory
✅ autodl环境检查通过

=== 分析现有数据文件 ===
数据文件: data/comprehensive_eval.jsonl
总数据量: 270 条
数据字段: ['abbr', 'english', 'chinese', 'instruction', 'input', 'output']
覆盖缩略语数量: X 个

=== 创建训练数据文件 ===
按缩略语分组后，共 X 个缩略语
训练集样本数: XXX
验证集样本数: XX

=== 创建数据集配置文件 ===
✅ 数据集配置文件已创建: data/dataset_info.json

=== 生成训练命令 ===
✅ 训练命令已保存到: correct_training_command.sh
```

## 🚨 **如果遇到问题**

### 问题1: 权限不足
```bash
chmod +x correct_training_command.sh
```

### 问题2: Python环境问题
```bash
# 确认环境
conda activate kb_enterprise_py310
python --version
```

### 问题3: 文件不存在
```bash
# 检查文件
ls -la data/
ls -la *.jsonl
```

## 🎯 **验证修复效果**

修复完成后，运行以下命令验证：

```bash
# 检查配置文件
cat data/dataset_info.json

# 检查训练数据
head -3 train_data_final.jsonl
head -3 eval_data_final.jsonl

# 尝试启动训练（测试配置）
python src/train.py \
  --model_name_or_path /root/autodl-tmp/enterprise_kb/models/transformers/DeepSeek-R1-Distill-Qwen-7B \
  --dataset company_abbreviations_train \
  --eval_dataset company_abbreviations_eval \
  --template qwen \
  --finetuning_type lora \
  --output_dir ./test_ckpt \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --max_steps 10 \
  --overwrite_output_dir
```

如果测试训练能正常启动，说明配置修复成功！

## 📞 **需要帮助？**

如果按照以上步骤仍然有问题，请提供：
1. `python fix_autodl_training.py` 的完整输出
2. `ls -la` 的目录列表
3. 具体的错误信息
