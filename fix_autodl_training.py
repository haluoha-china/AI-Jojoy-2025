#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autodl环境训练配置修复脚本
专门针对autodl上的企业知识库项目
"""

import json
import os
import random
from pathlib import Path

def check_autodl_environment():
    """检查autodl环境状态"""
    print("=== autodl环境检查 ===")
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查是否在正确的LLaMA-Factory目录
    if "LLaMA-Factory" not in current_dir:
        print("⚠️  警告: 当前不在LLaMA-Factory目录中")
        print("请确保在 /root/autodl-tmp/enterprise_kb/LLaMA-Factory 目录下运行此脚本")
        return False
    
    # 检查关键文件
    key_files = [
        "src/train.py",
        "data/comprehensive_eval.jsonl"
    ]
    
    missing_files = []
    for file_path in key_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少关键文件: {missing_files}")
        return False
    
    print("✅ autodl环境检查通过")
    return True

def analyze_comprehensive_eval():
    """分析comprehensive_eval.jsonl文件"""
    print("\n=== 分析现有数据文件 ===")
    
    eval_file = "data/comprehensive_eval.jsonl"
    
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"数据文件: {eval_file}")
        print(f"总数据量: {len(data)} 条")
        
        # 分析数据结构
        if data:
            sample = data[0]
            print(f"数据字段: {list(sample.keys())}")
            
            # 统计缩略语数量
            abbrs = set(item.get('abbr', '') for item in data if item.get('abbr'))
            print(f"覆盖缩略语数量: {len(abbrs)} 个")
            
            # 显示前几个缩略语
            print(f"示例缩略语: {list(abbrs)[:10]}")
        
        return data
        
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        return None

def create_training_files(data):
    """创建训练和验证文件"""
    print("\n=== 创建训练数据文件 ===")
    
    if not data:
        print("❌ 没有数据可以处理")
        return False
    
    # 按缩略语分组
    abbr_groups = {}
    for item in data:
        abbr = item.get('abbr', '')
        if abbr not in abbr_groups:
            abbr_groups[abbr] = []
        abbr_groups[abbr].append(item)
    
    print(f"按缩略语分组后，共 {len(abbr_groups)} 个缩略语")
    
    # 分割训练集和验证集 (80% 训练, 20% 验证)
    train_data = []
    eval_data = []
    
    for abbr, items in abbr_groups.items():
        # 随机打乱该缩略语的所有样本
        random.shuffle(items)
        
        # 计算分割点
        split_point = max(1, int(len(items) * 0.8))  # 确保每个缩略语至少有一个训练样本
        
        # 添加到训练集
        train_data.extend(items[:split_point])
        # 添加到验证集
        eval_data.extend(items[split_point:])
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(eval_data)}")
    
    # 保存训练集
    train_file = "train_data_final.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证集
    eval_file = "eval_data_final.jsonl"
    with open(eval_file, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 训练集已保存到: {train_file}")
    print(f"✅ 验证集已保存到: {eval_file}")
    
    return True

def create_dataset_info():
    """创建dataset_info.json配置文件"""
    print("\n=== 创建数据集配置文件 ===")
    
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
    config_file = "data/dataset_info.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据集配置文件已创建: {config_file}")
    
    # 显示配置内容
    print("\n配置文件内容:")
    print(json.dumps(config, ensure_ascii=False, indent=2))

def generate_training_command():
    """生成正确的训练命令"""
    print("\n=== 生成训练命令 ===")
    
    command = """python src/train.py \\
  --model_name_or_path /root/autodl-tmp/enterprise_kb/models/transformers/DeepSeek-R1-Distill-Qwen-7B \\
  --dataset company_abbreviations_train \\
  --eval_dataset company_abbreviations_eval \\
  --template qwen \\
  --finetuning_type lora \\
  --lora_rank 8 \\
  --lora_alpha 16 \\
  --lora_dropout 0.1 \\
  --lora_target q_proj,v_proj \\
  --output_dir ./lora_ckpt_prod \\
  --num_train_epochs 2 \\
  --per_device_train_batch_size 1 \\
  --gradient_accumulation_steps 16 \\
  --learning_rate 5e-5 \\
  --cutoff_len 256 \\
  --save_strategy steps \\
  --save_steps 100 \\
  --logging_steps 1 \\
  --overwrite_output_dir \\
  --bf16 true \\
  --metric_for_best_model eval_loss \\
  --load_best_model_at_end true"""
    
    print("正确的训练命令:")
    print("=" * 80)
    print(command)
    print("=" * 80)
    
    # 保存到文件
    with open("correct_training_command.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# 正确的训练命令 - autodl环境\n")
        f.write(command + "\n")
    
    print("\n✅ 训练命令已保存到: correct_training_command.sh")
    print("   可以直接运行: bash correct_training_command.sh")

def main():
    """主函数"""
    print("🚀 autodl环境训练配置修复脚本")
    print("=" * 50)
    
    # 检查环境
    if not check_autodl_environment():
        print("\n❌ 环境检查失败，请确保在正确的目录下运行")
        return
    
    try:
        # 分析现有数据
        data = analyze_comprehensive_eval()
        if not data:
            return
        
        # 创建训练文件
        if not create_training_files(data):
            return
        
        # 创建配置文件
        create_dataset_info()
        
        # 生成训练命令
        generate_training_command()
        
        print("\n🎉 修复完成！")
        print("\n下一步操作:")
        print("1. 清理输出目录: rm -rf ./lora_ckpt_prod")
        print("2. 运行训练命令: bash correct_training_command.sh")
        print("3. 或者直接复制上面的训练命令到终端执行")
        
    except Exception as e:
        print(f"\n❌ 修复过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
