#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试集生成器 - 分层验证策略
"""

import pandas as pd
import json
import random
from pathlib import Path

class TestGenerator:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.abbr_data = {}
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
                
                if abbr and cn:
                    self.abbr_data[abbr] = {
                        'english': eng,
                        'chinese': cn
                    }
            
            print(f"成功解析 {len(self.abbr_data)} 个有效缩略语")
            
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            raise
    
    def generate_basic_tests(self):
        """第一层：基础知识掌握度测试"""
        test_cases = []
        for abbr, info in self.abbr_data.items():
            test_cases.append({
                "test_id": f"basic_{abbr}_001",
                "layer": "基础知识掌握度测试",
                "instruction": f"请解释 '{abbr}' 的含义",
                "expected_answer": info['chinese'],
                "difficulty": "基础"
            })
        return test_cases
    
    def generate_confusion_tests(self):
        """第二层：混淆和边界测试"""
        test_cases = []
        abbr_list = list(self.abbr_data.keys())
        
        # 相似前缀测试
        for i, abbr1 in enumerate(abbr_list):
            for j, abbr2 in enumerate(abbr_list[i+1:], i+1):
                if len(abbr1) >= 2 and len(abbr2) >= 2:
                    if (abbr1[:2] == abbr2[:2] or abbr1[:3] == abbr2[:3]):
                        test_cases.append({
                            "test_id": f"confusion_{abbr1}_{abbr2}_001",
                            "layer": "混淆和边界测试",
                            "instruction": f"请区分 '{abbr1}' 和 '{abbr2}' 的含义",
                            "expected_answer": f"能正确区分 {abbr1} 和 {abbr2}",
                            "difficulty": "中等"
                        })
        
        return test_cases
    
    def generate_real_world_tests(self):
        """第三层：真实场景模拟测试"""
        test_cases = []
        
        # 业务场景测试
        scenarios = [
            "我们公司的{abbr}项目什么时候启动？",
            "请帮我找一下{abbr}相关的文档",
            "{abbr}系统出现故障，需要技术支持"
        ]
        
        for abbr in random.sample(list(self.abbr_data.keys()), min(20, len(self.abbr_data))):
            scenario = random.choice(scenarios)
            test_cases.append({
                "test_id": f"real_{abbr}_scenario_001",
                "layer": "真实场景模拟测试",
                "instruction": scenario.format(abbr=abbr),
                "expected_answer": f"能理解{abbr}的含义并在业务场景中应用",
                "difficulty": "高级"
            })
        
        # 负样本测试
        fake_abbrs = ["XYZ", "ABC", "DEF"]
        for fake_abbr in fake_abbrs:
            test_cases.append({
                "test_id": f"real_fake_{fake_abbr}_001",
                "layer": "真实场景模拟测试",
                "instruction": f"请解释 '{fake_abbr}' 的含义",
                "expected_answer": "应该表示不知道或无法识别",
                "difficulty": "高级"
            })
        
        return test_cases
    
    def generate_comprehensive_test_set(self):
        """生成完整的综合测试集"""
        print("开始生成综合测试集...")
        
        layer1_tests = self.generate_basic_tests()
        layer2_tests = self.generate_confusion_tests()
        layer3_tests = self.generate_real_world_tests()
        
        all_tests = layer1_tests + layer2_tests + layer3_tests
        
        test_set = {
            "metadata": {
                "description": "企业知识库缩略语综合测试集 - 分层验证策略",
                "version": "1.0",
                "statistics": {
                    "total_test_cases": len(all_tests),
                    "layer1_basic": len(layer1_tests),
                    "layer2_confusion": len(layer2_tests),
                    "layer3_real_world": len(layer3_tests)
                }
            },
            "test_cases": all_tests
        }
        
        print(f"综合测试集生成完成！总计: {len(all_tests)} 个测试用例")
        return test_set
    
    def save_test_set(self, test_set, output_path=None):
        """保存测试集"""
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"comprehensive_test_set_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
        
        print(f"测试集已保存到: {output_path}")
        return output_path

def main():
    print("=== 企业知识库综合测试集生成器 ===")
    
    excel_path = "公司常用缩略语20250401.xlsx"
    
    try:
        generator = TestGenerator(excel_path)
        test_set = generator.generate_comprehensive_test_set()
        json_path = generator.save_test_set(test_set)
        
        print(f"\n生成完成！文件保存为: {json_path}")
        print("\n您现在可以：")
        print("1. 继续使用完整数据进行模型训练")
        print("2. 使用这个综合测试集验证模型效果")
        
    except Exception as e:
        print(f"生成测试集时发生错误: {e}")

if __name__ == "__main__":
    main()
