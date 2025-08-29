#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控脚本 - 实时监控全量数据训练进度
"""

import os
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, output_dir: str = "./lora_ckpt_full_data"):
        self.output_dir = output_dir
        self.start_time = datetime.now()
        self.last_checkpoint = None
        
    def check_gpu_status(self):
        """检查GPU状态"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    memory_used, memory_total, gpu_util = line.split(', ')
                    print(f"GPU {i}: 显存使用 {memory_used}/{memory_total}MB, 利用率 {gpu_util}%")
        except Exception as e:
            print(f"无法获取GPU状态: {e}")
    
    def check_training_progress(self):
        """检查训练进度"""
        if not os.path.exists(self.output_dir):
            print("❌ 训练输出目录不存在")
            return False
        
        # 查找最新的检查点
        checkpoints = []
        for item in os.listdir(self.output_dir):
            if item.startswith('checkpoint-'):
                try:
                    step = int(item.split('-')[1])
                    checkpoints.append((step, item))
                except:
                    continue
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x[0])
            if latest_checkpoint != self.last_checkpoint:
                self.last_checkpoint = latest_checkpoint
                print(f"✅ 发现新检查点: {latest_checkpoint[1]} (步骤 {latest_checkpoint[0]})")
            
            # 检查训练日志
            log_file = os.path.join(self.output_dir, 'trainer_state.json')
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        state = json.load(f)
                    if 'global_step' in state:
                        print(f"📊 当前训练步数: {state['global_step']}")
                    if 'epoch' in state:
                        print(f"📈 当前训练轮数: {state['epoch']:.2f}")
                except:
                    pass
        
        return True
    
    def check_data_files(self):
        """检查数据文件状态"""
        data_files = [
            "train_data_final.jsonl",
            "eval_data_final.jsonl", 
            "data/dataset_info.json"
        ]
        
        print("\n📁 数据文件状态:")
        for file_path in data_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if file_path.endswith('.jsonl'):
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    print(f"  ✅ {file_path}: {size} bytes, {line_count} 行")
                else:
                    print(f"  ✅ {file_path}: {size} bytes")
            else:
                print(f"  ❌ {file_path}: 文件不存在")
    
    def display_training_info(self):
        """显示训练信息"""
        print("\n" + "="*60)
        print("企业知识库全量数据训练监控")
        print("="*60)
        print(f"监控开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"训练输出目录: {self.output_dir}")
        print(f"训练目标: 确保所有264个缩略语都被完整学习")
        print("="*60)
    
    def run_monitoring(self, interval: int = 30):
        """运行监控循环"""
        self.display_training_info()
        
        try:
            while True:
                print(f"\n🕐 {datetime.now().strftime('%H:%M:%S')} - 训练状态检查")
                print("-" * 40)
                
                # 检查GPU状态
                print("🔍 GPU状态:")
                self.check_gpu_status()
                
                # 检查训练进度
                print("\n🔍 训练进度:")
                if not self.check_training_progress():
                    print("⚠️  训练可能未启动或已停止")
                
                # 检查数据文件
                self.check_data_files()
                
                print(f"\n⏳ {interval}秒后进行下次检查...")
                print("按 Ctrl+C 停止监控")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 监控已停止")
            print("训练监控完成！")

def main():
    """主函数"""
    print("=== 企业知识库训练监控器 ===")
    
    # 检查是否在LLaMA-Factory目录
    if not os.path.exists("src/train.py"):
        print("❌ 请在LLaMA-Factory目录下运行此脚本")
        return
    
    # 创建监控器
    monitor = TrainingMonitor()
    
    # 开始监控
    monitor.run_monitoring()

if __name__ == "__main__":
    main()
