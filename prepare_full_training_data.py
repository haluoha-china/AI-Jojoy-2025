#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量数据准备脚本 - 基于DeepSeek分层验证策略
确保所有缩略语都被包含在训练数据中
"""

import pandas as pd
import json
import random
from pathlib import Path
from typing import Dict, List, Any

class FullTrainingDataPreparer:
    """全量训练数据准备器"""
    
    def __init__(self, excel_path: str):
        """
        初始化数据准备器
        
        Args:
            excel_path: Excel文件路径
        """
        self.excel_path = excel_path
        self.abbr_data = {}
        self.training_data = []
        self.load_data()
        
    def load_data(self):
        """加载Excel数据"""
        try:
            df = pd.read_excel(self.excel_path)
            print(f"成功读取Excel文件，共 {len(df)} 行数据")
            
            for _, row in df.iterrows():
                abbr = str(row.iloc[0]).strip().upper()
                eng = str(row.iloc[1]).strip()
                cn = str(row.iloc[2]).strip()
                
                if abbr and cn:  # 确保数据有效
                    self.abbr_data[abbr] = {
                        'english': eng,
                        'chinese': cn,
                        'full_name': f"{eng} ({cn})"
                    }
            
            print(f"成功解析 {len(self.abbr_data)} 个有效缩略语")
            
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            raise
    
    def generate_basic_instruction_data(self) -> List[Dict[str, str]]:
        """生成基础指令数据 - 第一层测试对应"""
        print("生成基础指令数据...")
        
        data = []
        for abbr, info in self.abbr_data.items():
            # 基础解释指令
            data.append({
                "instruction": f"请解释 '{abbr}' 的含义",
                "input": "",
                "output": f"{abbr} 是 {info['english']} 的缩写，中文含义是：{info['chinese']}。"
            })
            
            # 英文全称指令
            if info['english']:
                data.append({
                    "instruction": f"'{abbr}' 的英文全称是什么？",
                    "input": "",
                    "output": f"{abbr} 的英文全称是：{info['english']}。"
                })
            
            # 中文含义指令
            data.append({
                "instruction": f"'{abbr}' 的中文含义是什么？",
                "input": "",
                "output": f"{abbr} 的中文含义是：{info['chinese']}。"
            })
        
        print(f"基础指令数据生成完成，共 {len(data)} 条")
        return data
    
    def generate_confusion_instruction_data(self) -> List[Dict[str, str]]:
        """生成混淆区分指令数据 - 第二层测试对应"""
        print("生成混淆区分指令数据...")
        
        data = []
        abbr_list = list(self.abbr_data.keys())
        
        # 相似前缀缩略语区分
        for i, abbr1 in enumerate(abbr_list):
            for j, abbr2 in enumerate(abbr_list[i+1:], i+1):
                if len(abbr1) >= 2 and len(abbr2) >= 2:
                    # 检查是否有相似前缀
                    if (abbr1[:2] == abbr2[:2] or 
                        abbr1[:3] == abbr2[:3] or
                        abbr1[-2:] == abbr2[-2:]):
                        
                        data.append({
                            "instruction": f"请区分 '{abbr1}' 和 '{abbr2}' 的含义，它们有什么不同？",
                            "input": "",
                            "output": f"{abbr1} 是 {self.abbr_data[abbr1]['english']} 的缩写，中文含义是：{self.abbr_data[abbr1]['chinese']}。{abbr2} 是 {self.abbr_data[abbr2]['english']} 的缩写，中文含义是：{self.abbr_data[abbr2]['chinese']}。虽然它们有相似的前缀或后缀，但代表不同的概念和功能。"
                        })
        
        # 上下文区分指令
        context_scenarios = [
            "在技术领域",
            "在管理领域", 
            "在业务领域",
            "在财务领域",
            "在人力资源领域"
        ]
        
        for abbr in random.sample(list(self.abbr_data.keys()), min(30, len(self.abbr_data))):
            context = random.choice(context_scenarios)
            data.append({
                "instruction": f"{context}，'{abbr}' 通常指什么？",
                "input": "",
                "output": f"{context}，{abbr} 是 {self.abbr_data[abbr]['english']} 的缩写，中文含义是：{self.abbr_data[abbr]['chinese']}。"
            })
        
        print(f"混淆区分指令数据生成完成，共 {len(data)} 条")
        return data
    
    def generate_real_world_instruction_data(self) -> List[Dict[str, str]]:
        """生成真实场景指令数据 - 第三层测试对应"""
        print("生成真实场景指令数据...")
        
        data = []
        
        # 业务场景指令
        business_scenarios = [
            "我们公司的{abbr}项目什么时候启动？",
            "请帮我找一下{abbr}相关的文档",
            "{abbr}系统出现故障，需要技术支持",
            "我想了解{abbr}的具体流程",
            "请评估一下{abbr}的风险等级",
            "{abbr}认证需要准备哪些材料？",
            "我们是否需要升级{abbr}？",
            "请分析{abbr}的成本效益",
            "{abbr}培训什么时候开始？",
            "我想申请{abbr}的权限"
        ]
        
        for abbr in random.sample(list(self.abbr_data.keys()), min(40, len(self.abbr_data))):
            scenario = random.choice(business_scenarios)
            data.append({
                "instruction": scenario.format(abbr=abbr),
                "input": "",
                "output": f"关于{abbr}（{self.abbr_data[abbr]['english']}，{self.abbr_data[abbr]['chinese']}）的问题，我需要更多具体信息来帮助您。{abbr}是我们公司的重要系统/流程，具体操作请参考相关文档或联系相关部门。"
            })
        
        # 复合缩略语理解指令
        for _ in range(min(25, len(self.abbr_data) // 3)):
            selected_abbrs = random.sample(list(self.abbr_data.keys()), random.randint(2, 3))
            combined_question = f"请解释一下 {', '.join(selected_abbrs)} 这几个缩略语的含义和它们之间的关系"
            
            output_parts = []
            for abbr in selected_abbrs:
                output_parts.append(f"{abbr} 是 {self.abbr_data[abbr]['english']} 的缩写，中文含义是：{self.abbr_data[abbr]['chinese']}")
            
            data.append({
                "instruction": combined_question,
                "input": "",
                "output": "这几个缩略语的含义如下：" + "；".join(output_parts) + "。它们都是公司重要的业务系统或流程，相互配合支持公司的日常运营。"
            })
        
        # 负样本指令（测试模型拒绝能力）
        fake_abbrs = ["XYZ", "ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX"]
        for fake_abbr in fake_abbrs:
            data.append({
                "instruction": f"请解释 '{fake_abbr}' 的含义",
                "input": "",
                "output": f"抱歉，'{fake_abbr}' 不是我们公司的标准缩略语。如果您需要了解某个缩略语的含义，请提供正确的缩略语，我会很乐意为您解释。"
            })
        
        print(f"真实场景指令数据生成完成，共 {len(data)} 条")
        return data
    
    def generate_enhanced_training_data(self) -> List[Dict[str, str]]:
        """生成增强训练数据"""
        print("生成增强训练数据...")
        
        data = []
        
        # 1. 基础指令数据
        data.extend(self.generate_basic_instruction_data())
        
        # 2. 混淆区分指令数据
        data.extend(self.generate_confusion_instruction_data())
        
        # 3. 真实场景指令数据
        data.extend(self.generate_real_world_instruction_data())
        
        # 4. 额外增强数据
        # 同义词和近义词训练
        synonyms = {
            "系统": ["平台", "软件", "应用"],
            "管理": ["控制", "监督", "协调"],
            "服务": ["支持", "协助", "帮助"],
            "流程": ["程序", "步骤", "方法"],
            "标准": ["规范", "要求", "准则"]
        }
        
        for abbr in random.sample(list(self.abbr_data.keys()), min(20, len(self.abbr_data))):
            for main_word, syn_list in synonyms.items():
                if main_word in self.abbr_data[abbr]['chinese']:
                    for syn in syn_list:
                        data.append({
                            "instruction": f"'{abbr}' 可以用什么词来描述？",
                            "input": "",
                            "output": f"{abbr} 是 {self.abbr_data[abbr]['english']} 的缩写，可以用 '{syn}' 来描述，中文含义是：{self.abbr_data[abbr]['chinese']}。"
                        })
        
        print(f"增强训练数据生成完成，总计 {len(data)} 条")
        return data
    
    def save_training_data(self, data: List[Dict[str, str]], output_path: str = None):
        """保存训练数据"""
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"full_training_data_{timestamp}.jsonl"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"训练数据已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"保存训练数据失败: {e}")
            raise
    
    def generate_training_summary(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成训练数据摘要"""
        # 统计指令类型
        instruction_types = {}
        for item in data:
            instruction = item['instruction']
            if "请解释" in instruction and "的含义" in instruction:
                instruction_types['基础解释'] = instruction_types.get('基础解释', 0) + 1
            elif "英文全称" in instruction:
                instruction_types['英文全称'] = instruction_types.get('英文全称', 0) + 1
            elif "区分" in instruction:
                instruction_types['混淆区分'] = instruction_types.get('混淆区分', 0) + 1
            elif "在" in instruction and "领域" in instruction:
                instruction_types['上下文区分'] = instruction_types.get('上下文区分', 0) + 1
            elif any(scenario in instruction for scenario in ["项目", "系统", "故障", "流程", "培训"]):
                instruction_types['业务场景'] = instruction_types.get('业务场景', 0) + 1
            elif "几个缩略语" in instruction:
                instruction_types['复合理解'] = instruction_types.get('复合理解', 0) + 1
            elif any(fake in instruction for fake in ["XYZ", "ABC", "DEF"]):
                instruction_types['负样本'] = instruction_types.get('负样本', 0) + 1
            else:
                instruction_types['其他'] = instruction_types.get('其他', 0) + 1
        
        summary = {
            "total_instructions": len(data),
            "abbreviations_covered": len(self.abbr_data),
            "instruction_type_distribution": instruction_types,
            "generation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        return summary
    
    def print_training_summary(self, summary: Dict[str, Any]):
        """打印训练数据摘要"""
        print("\n" + "="*60)
        print("全量训练数据生成摘要")
        print("="*60)
        print(f"总指令数量: {summary['total_instructions']}")
        print(f"覆盖缩略语数量: {summary['abbreviations_covered']}")
        print(f"\n指令类型分布:")
        for inst_type, count in summary['instruction_type_distribution'].items():
            print(f"  {inst_type}: {count} 条")
        print("="*60)

def main():
    """主函数"""
    print("=== 企业知识库全量训练数据准备器 ===")
    print("基于DeepSeek分层验证策略")
    print()
    
    # 设置Excel文件路径
    excel_path = "公司常用缩略语20250401.xlsx"
    
    try:
        # 创建数据准备器
        preparer = FullTrainingDataPreparer(excel_path)
        
        # 生成增强训练数据
        training_data = preparer.generate_enhanced_training_data()
        
        # 保存训练数据
        jsonl_path = preparer.save_training_data(training_data)
        
        # 生成并打印摘要
        summary = preparer.generate_training_summary(training_data)
        preparer.print_training_summary(summary)
        
        print(f"\n训练数据已保存到: {jsonl_path}")
        print("\n您现在可以：")
        print("1. 使用这个全量训练数据进行模型训练")
        print("2. 确保所有缩略语都被完整学习")
        print("3. 训练完成后使用分层验证测试集验证效果")
        
    except Exception as e:
        print(f"准备训练数据时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
