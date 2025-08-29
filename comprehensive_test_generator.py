#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试集生成器 - 实现分层验证策略
基于DeepSeek的优化建议设计

作者：AI助手
日期：2025-01-XX
"""

import pandas as pd
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

class ComprehensiveTestGenerator:
    """综合测试集生成器 - 分层验证策略"""
    
    def __init__(self, excel_path: str):
        """
        初始化测试生成器
        
        Args:
            excel_path: Excel文件路径
        """
        self.excel_path = excel_path
        self.abbr_data = {}
        self.test_sets = {}
        self.load_data()
        
    def load_data(self):
        """加载Excel数据"""
        try:
            df = pd.read_excel(self.excel_path)
            print(f"成功读取Excel文件，共 {len(df)} 行数据")
            
            # 建立缩写映射
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
    
    def generate_layer1_basic_knowledge_tests(self) -> List[Dict]:
        """
        第一层：基础知识掌握度测试
        测试模型对所有已知缩略语的记忆准确度
        """
        print("生成第一层测试：基础知识掌握度测试...")
        
        test_cases = []
        for abbr, info in self.abbr_data.items():
            # 基础解释测试
            test_cases.append({
                "test_id": f"basic_{abbr}_001",
                "layer": "基础知识掌握度测试",
                "category": "基础解释",
                "instruction": f"请解释 '{abbr}' 的含义",
                "expected_answer": info['chinese'],
                "expected_keywords": [info['chinese'], info['english']],
                "difficulty": "基础",
                "description": f"测试模型对缩略语 {abbr} 的基础理解"
            })
            
            # 英文全称测试
            if info['english']:
                test_cases.append({
                    "test_id": f"basic_{abbr}_002",
                    "layer": "基础知识掌握度测试",
                    "category": "英文全称",
                    "instruction": f"'{abbr}' 的英文全称是什么？",
                    "expected_answer": info['english'],
                    "expected_keywords": [info['english']],
                    "difficulty": "基础",
                    "description": f"测试模型对缩略语 {abbr} 英文全称的掌握"
                })
        
        print(f"第一层测试生成完成，共 {len(test_cases)} 个测试用例")
        return test_cases
    
    def generate_layer2_confusion_boundary_tests(self) -> List[Dict]:
        """
        第二层：混淆和边界测试
        测试模型对相似缩略语的区分能力
        """
        print("生成第二层测试：混淆和边界测试...")
        
        test_cases = []
        
        # 1. 相似前缀测试
        abbr_list = list(self.abbr_data.keys())
        for i, abbr1 in enumerate(abbr_list):
            for j, abbr2 in enumerate(abbr_list[i+1:], i+1):
                if len(abbr1) >= 2 and len(abbr2) >= 2:
                    # 检查是否有相似前缀
                    if (abbr1[:2] == abbr2[:2] or 
                        abbr1[:3] == abbr2[:3] or
                        abbr1[-2:] == abbr2[-2:]):
                        
                        test_cases.append({
                            "test_id": f"confusion_{abbr1}_{abbr2}_001",
                            "layer": "混淆和边界测试",
                            "category": "相似前缀区分",
                            "instruction": f"请区分 '{abbr1}' 和 '{abbr2}' 的含义，它们有什么不同？",
                            "expected_answer": f"能正确区分 {abbr1} 和 {abbr2}",
                            "expected_keywords": [self.abbr_data[abbr1]['chinese'], 
                                                self.abbr_data[abbr2]['chinese']],
                            "difficulty": "中等",
                            "description": f"测试模型对相似缩略语 {abbr1} 和 {abbr2} 的区分能力"
                        })
        
        # 2. 上下文区分测试
        # 为一些常见缩略语创建不同上下文的测试
        common_contexts = {
            "技术": ["系统", "平台", "软件", "硬件", "网络"],
            "管理": ["流程", "制度", "标准", "规范", "体系"],
            "业务": ["产品", "服务", "客户", "市场", "销售"]
        }
        
        for abbr in random.sample(list(self.abbr_data.keys()), min(20, len(self.abbr_data))):
            for context_type, contexts in common_contexts.items():
                context = random.choice(contexts)
                test_cases.append({
                    "test_id": f"context_{abbr}_{context_type}_001",
                    "layer": "混淆和边界测试",
                    "category": "上下文区分",
                    "instruction": f"在{context_type}领域，'{abbr}' 通常指什么？",
                    "expected_answer": f"能根据{context_type}上下文正确解释 {abbr}",
                    "expected_keywords": [self.abbr_data[abbr]['chinese']],
                    "difficulty": "中等",
                    "description": f"测试模型在{context_type}上下文中对 {abbr} 的理解"
                })
        
        print(f"第二层测试生成完成，共 {len(test_cases)} 个测试用例")
        return test_cases
    
    def generate_layer3_real_world_tests(self) -> List[Dict]:
        """
        第三层：真实场景模拟测试
        模拟真实用户提问，测试在真实语境中的表现
        """
        print("生成第三层测试：真实场景模拟测试...")
        
        test_cases = []
        
        # 1. 业务场景测试
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
        
        for abbr in random.sample(list(self.abbr_data.keys()), min(30, len(self.abbr_data))):
            scenario = random.choice(business_scenarios)
            test_cases.append({
                "test_id": f"real_{abbr}_scenario_001",
                "layer": "真实场景模拟测试",
                "category": "业务场景",
                "instruction": scenario.format(abbr=abbr),
                "expected_answer": f"能理解{abbr}的含义并在业务场景中正确应用",
                "expected_keywords": [self.abbr_data[abbr]['chinese']],
                "difficulty": "高级",
                "description": f"测试模型在真实业务场景中对 {abbr} 的理解和应用"
            })
        
        # 2. 复合缩略语测试
        # 随机选择2-3个缩略语组合提问
        for _ in range(min(20, len(self.abbr_data) // 3)):
            selected_abbrs = random.sample(list(self.abbr_data.keys()), random.randint(2, 3))
            combined_question = f"请解释一下 {', '.join(selected_abbrs)} 这几个缩略语的含义和它们之间的关系"
            
            test_cases.append({
                "test_id": f"real_combined_{'_'.join(selected_abbrs)}_001",
                "layer": "真实场景模拟测试",
                "category": "复合理解",
                "instruction": combined_question,
                "expected_answer": f"能正确解释所有缩略语并分析它们的关系",
                "expected_keywords": [self.abbr_data[abbr]['chinese'] for abbr in selected_abbrs],
                "difficulty": "高级",
                "description": f"测试模型对多个缩略语的复合理解能力"
            })
        
        # 3. 负样本测试（测试模型会不会胡乱回答）
        fake_abbrs = ["XYZ", "ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX"]
        for fake_abbr in fake_abbrs:
            test_cases.append({
                "test_id": f"real_fake_{fake_abbr}_001",
                "layer": "真实场景模拟测试",
                "category": "负样本测试",
                "instruction": f"请解释 '{fake_abbr}' 的含义",
                "expected_answer": "应该表示不知道或无法识别",
                "expected_keywords": ["不知道", "无法识别", "不存在", "未定义"],
                "difficulty": "高级",
                "description": f"测试模型对不存在的缩略语 {fake_abbr} 的正确拒绝能力"
            })
        
        print(f"第三层测试生成完成，共 {len(test_cases)} 个测试用例")
        return test_cases
    
    def generate_comprehensive_test_set(self) -> Dict[str, Any]:
        """
        生成完整的综合测试集
        """
        print("开始生成综合测试集...")
        
        # 生成三层测试
        layer1_tests = self.generate_layer1_basic_knowledge_tests()
        layer2_tests = self.generate_layer2_confusion_boundary_tests()
        layer3_tests = self.generate_layer3_real_world_tests()
        
        # 合并所有测试
        all_tests = layer1_tests + layer2_tests + layer3_tests
        
        # 创建测试集统计
        test_statistics = {
            "total_test_cases": len(all_tests),
            "layer1_basic_knowledge": len(layer1_tests),
            "layer2_confusion_boundary": len(layer2_tests),
            "layer3_real_world": len(layer3_tests),
            "abbreviations_covered": len(self.abbr_data),
            "generation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 创建完整的测试集
        comprehensive_test_set = {
            "metadata": {
                "description": "企业知识库缩略语综合测试集 - 分层验证策略",
                "version": "1.0",
                "author": "AI助手",
                "based_on": "DeepSeek优化建议",
                "statistics": test_statistics
            },
            "test_cases": all_tests,
            "abbreviations_reference": self.abbr_data
        }
        
        print(f"综合测试集生成完成！")
        print(f"总计: {len(all_tests)} 个测试用例")
        print(f"第一层: {len(layer1_tests)} 个")
        print(f"第二层: {len(layer2_tests)} 个") 
        print(f"第三层: {len(layer3_tests)} 个")
        
        return comprehensive_test_set
    
    def save_test_set(self, test_set: Dict[str, Any], output_path: str = None):
        """
        保存测试集到文件
        
        Args:
            test_set: 测试集数据
            output_path: 输出文件路径
        """
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"comprehensive_test_set_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_set, f, ensure_ascii=False, indent=2)
            
            print(f"测试集已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"保存测试集失败: {e}")
            raise
    
    def export_test_cases_to_excel(self, test_set: Dict[str, Any], output_path: str = None):
        """
        将测试用例导出到Excel文件，便于人工查看和编辑
        
        Args:
            test_set: 测试集数据
            output_path: 输出Excel文件路径
        """
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"comprehensive_test_cases_{timestamp}.xlsx"
        
        try:
            # 准备数据
            test_data = []
            for test_case in test_set['test_cases']:
                test_data.append({
                    "测试ID": test_case['test_id'],
                    "测试层级": test_case['layer'],
                    "测试类别": test_case['category'],
                    "测试指令": test_case['instruction'],
                    "期望答案": test_case['expected_answer'],
                    "期望关键词": ", ".join(test_case['expected_keywords']),
                    "难度等级": test_case['difficulty'],
                    "测试描述": test_case['description']
                })
            
            # 创建DataFrame并保存
            df = pd.DataFrame(test_data)
            df.to_excel(output_path, index=False, sheet_name='综合测试用例')
            
            print(f"测试用例已导出到Excel: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"导出Excel失败: {e}")
            raise

def main():
    """主函数"""
    print("=== 企业知识库综合测试集生成器 ===")
    print("基于DeepSeek的分层验证策略优化建议")
    print()
    
    # 设置Excel文件路径
    excel_path = "公司常用缩略语20250401.xlsx"
    
    try:
        # 创建测试生成器
        generator = ComprehensiveTestGenerator(excel_path)
        
        # 生成综合测试集
        test_set = generator.generate_comprehensive_test_set()
        
        # 保存为JSON格式
        json_path = generator.save_test_set(test_set)
        
        # 导出为Excel格式（便于查看）
        excel_path = generator.export_test_cases_to_excel(test_set)
        
        print("\n=== 生成完成 ===")
        print(f"JSON格式: {json_path}")
        print(f"Excel格式: {excel_path}")
        print("\n您现在可以：")
        print("1. 继续使用完整数据进行模型训练")
        print("2. 使用这个综合测试集验证模型效果")
        print("3. 根据测试结果调整训练策略")
        
    except Exception as e:
        print(f"生成测试集时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
