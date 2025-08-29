#!/bin/bash
# 全量数据训练命令 - 修正配置问题
# 基于DeepSeek分层验证策略

echo "=== 企业知识库全量数据训练 ==="
echo "基于DeepSeek分层验证策略"
echo "确保所有264个缩略语都被完整学习"
echo ""

# 检查数据集配置
echo "1. 验证数据集配置..."
if [ ! -f "data/dataset_info.json" ]; then
    echo "❌ 数据集配置文件不存在"
    exit 1
fi

if [ ! -f "train_data_final.jsonl" ] || [ ! -f "eval_data_final.jsonl" ]; then
    echo "❌ 训练数据文件不存在"
    exit 1
fi

echo "✅ 数据集配置验证通过"
echo ""

# 显示数据统计
echo "2. 数据统计信息:"
echo "   训练集样本数: $(wc -l < train_data_final.jsonl)"
echo "   验证集样本数: $(wc -l < eval_data_final.jsonl)"
echo "   覆盖缩略语数: $(cat train_data_final.jsonl | grep -o '"abbr": "[^"]*"' | sort | uniq | wc -l)"
echo ""

# 开始训练
echo "3. 启动全量数据训练..."
echo "   训练目标: 确保所有264个缩略语都被完整学习"
echo "   训练策略: 分层验证 + 充分训练"
echo ""

python src/train.py \
  --model_name_or_path /root/autodl-tmp/enterprise_kb/models/transformers/DeepSeek-R1-Distill-Qwen-7B \
  --dataset "company_abbreviations_train,company_abbreviations_eval" \
  --template qwen \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir ./lora_ckpt_full_data \
  --num_train_epochs 20 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --max_source_length 512 \
  --max_target_length 128 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --fp16 \
  --max_grad_norm 1.0 \
  --logging_steps 50 \
  --overwrite_output_dir

echo ""
echo "4. 训练完成！"
echo "   检查点保存在: ./lora_ckpt_full_data"
echo "   下一步: 使用分层验证测试集验证效果"
