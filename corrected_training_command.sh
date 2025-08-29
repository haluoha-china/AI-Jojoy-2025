#!/bin/bash
# 修正后的训练命令 - 确保数据集配置正确
# 基于kimi的建议和实际配置

echo "=== 企业知识库训练命令修正版 ==="
echo "确保数据集配置与dataset_info.json一致"
echo ""

# 检查数据集配置文件
echo "1. 验证数据集配置..."
if [ ! -f "data/dataset_info.json" ]; then
    echo "❌ 数据集配置文件不存在，正在创建..."
    mkdir -p data
    cat > data/dataset_info.json << 'EOF'
{
  "company_abbreviations_train": {
    "file_name": "train_data_final.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "company_abbreviations_eval": {
    "file_name": "eval_data_final.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
EOF
    echo "✅ 数据集配置文件已创建"
else
    echo "✅ 数据集配置文件存在"
fi

# 检查训练数据文件
echo ""
echo "2. 检查训练数据文件..."
if [ ! -f "train_data_final.jsonl" ]; then
    echo "❌ 训练数据文件不存在: train_data_final.jsonl"
    echo "请先运行 prepare_full_training_data.py 生成训练数据"
    exit 1
fi

if [ ! -f "eval_data_final.jsonl" ]; then
    echo "❌ 验证数据文件不存在: eval_data_final.jsonl"
    echo "请先运行 prepare_full_training_data.py 生成验证数据"
    exit 1
fi

echo "✅ 训练数据文件检查通过"
echo "   训练集: $(wc -l < train_data_final.jsonl) 行"
echo "   验证集: $(wc -l < eval_data_final.jsonl) 行"

# 清理输出目录
echo ""
echo "3. 清理输出目录..."
if [ -d "./lora_ckpt_prod" ]; then
    rm -rf ./lora_ckpt_prod
    echo "✅ 输出目录已清理"
else
    echo "✅ 输出目录不存在，无需清理"
fi

# 显示正确的训练命令
echo ""
echo "4. 正确的训练命令:"
echo "=========================================="
echo "python src/train.py \\"
echo "  --model_name_or_path /root/autodl-tmp/enterprise_kb/models/transformers/DeepSeek-R1-Distill-Qwen-7B \\"
echo "  --dataset company_abbreviations_train \\"
echo "  --eval_dataset company_abbreviations_eval \\"
echo "  --template qwen \\"
echo "  --finetuning_type lora \\"
echo "  --lora_rank 8 \\"
echo "  --lora_alpha 16 \\"
echo "  --lora_dropout 0.1 \\"
echo "  --lora_target q_proj,v_proj \\"
echo "  --output_dir ./lora_ckpt_prod \\"
echo "  --num_train_epochs 2 \\"
echo "  --per_device_train_batch_size 1 \\"
echo "  --gradient_accumulation_steps 16 \\"
echo "  --learning_rate 5e-5 \\"
echo "  --cutoff_len 256 \\"
echo "  --save_strategy steps \\"
echo "  --save_steps 100 \\"
echo "  --logging_steps 1 \\"
echo "  --overwrite_output_dir \\"
echo "  --bf16 true \\"
echo "  --metric_for_best_model eval_loss \\"
echo "  --load_best_model_at_end true"
echo "=========================================="

echo ""
echo "5. 关键配置说明:"
echo "   ✅ --dataset: company_abbreviations_train (对应dataset_info.json中的键名)"
echo "   ✅ --eval_dataset: company_abbreviations_eval (对应dataset_info.json中的键名)"
echo "   ✅ 不是文件名，而是dataset_info.json中定义的键名"
echo ""
echo "6. 现在可以运行训练命令了！"
echo "   复制上面的命令到终端执行即可"
