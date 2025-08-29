#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据文件准备脚本
将现有的comprehensive_eval.jsonl转换为训练所需的格式
"""

import json
import os
from pathlib import Path

def convert_to_training_format(input_file, train_output, eval_output, train_ratio=0.8):
    """
    将comprehensive_eval.jsonl转换为训练和验证集
    
    Args:
        input_file: 输入的comprehensive_eval.jsonl文件
        train_output: 训练集输出文件
        eval_output: 验证集输出文件
        train_ratio: 训练集比例，默认0.8
    """
    
    print(f"正在读取文件: {input_file}")
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"总数据量: {len(data)} 条")
    
    # 按缩略语分组，确保每个缩略语都有训练和验证样本
    abbr_groups = {}
    for item in data:
        abbr = item.get('abbr', '')
        if abbr not in abbr_groups:
            abbr_groups[abbr] = []
        abbr_groups[abbr].append(item)
    
    print(f"覆盖缩略语数量: {len(abbr_groups)} 个")
    
    # 分割训练集和验证集
    train_data = []
    eval_data = []
    
    for abbr, items in abbr_groups.items():
        # 随机打乱该缩略语的所有样本
        import random
        random.shuffle(items)
        
        # 计算分割点
        split_point = int(len(items) * train_ratio)
        
        # 添加到训练集
        train_data.extend(items[:split_point])
        # 添加到验证集
        eval_data.extend(items[split_point:])
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(eval_data)}")
    
    # 保存训练集
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证集
    with open(eval_output, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 训练集已保存到: {train_output}")
    print(f"✅ 验证集已保存到: {eval_output}")
    
    # 验证数据完整性
    print("\n数据完整性验证:")
    print(f"  训练集覆盖缩略语: {len(set(item.get('abbr', '') for item in train_data))}")
    print(f"  验证集覆盖缩略语: {len(set(item.get('abbr', '') for item in eval_data))}")
    
    return True

def create_dataset_info():
    """创建dataset_info.json配置文件"""
    
    config = {
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
    
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    
    # 保存配置文件
    with open('data/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ dataset_info.json 配置文件已创建")

def main():
    """主函数"""
    print("=== 训练数据文件准备脚本 ===")
    print("将comprehensive_eval.jsonl转换为训练所需格式")
    print()
    
    # 检查输入文件
    input_file = "comprehensive_eval.jsonl"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        print("请确保comprehensive_eval.jsonl文件在当前目录")
        return
    
    # 输出文件名
    train_output = "train_data_final.jsonl"
    eval_output = "eval_data_final.jsonl"
    
    try:
        # 转换数据格式
        success = convert_to_training_format(input_file, train_output, eval_output)
        
        if success:
            # 创建数据集配置文件
            create_dataset_info()
            
            print("\n🎉 数据准备完成！")
            print("现在可以运行训练命令了:")
            print()
            print("python src/train.py \\")
            print("  --model_name_or_path /root/autodl-tmp/enterprise_kb/models/transformers/DeepSeek-R1-Distill-Qwen-7B \\")
            print("  --dataset company_abbreviations_train \\")
            print("  --eval_dataset company_abbreviations_eval \\")
            print("  --template qwen \\")
            print("  --finetuning_type lora \\")
            print("  --lora_rank 8 \\")
            print("  --lora_alpha 16 \\")
            print("  --lora_dropout 0.1 \\")
            print("  --lora_target q_proj,v_proj \\")
            print("  --output_dir ./lora_ckpt_prod \\")
            print("  --num_train_epochs 2 \\")
            print("  --per_device_train_batch_size 1 \\")
            print("  --gradient_accumulation_steps 16 \\")
            print("  --learning_rate 5e-5 \\")
            print("  --cutoff_len 256 \\")
            print("  --save_strategy steps \\")
            print("  --save_steps 100 \\")
            print("  --logging_steps 1 \\")
            print("  --overwrite_output_dir \\")
            print("  --bf16 true \\")
            print("  --metric_for_best_model eval_loss \\")
            print("  --load_best_model_at_end true")
        
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
