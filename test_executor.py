#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试执行器 - 执行综合测试集并评估模型效果
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import time
from datetime import datetime

class TestExecutor:
    """测试执行器"""
    
    def __init__(self, test_set_path: str):
        """
        初始化测试执行器
        
        Args:
            test_set_path: 测试集文件路径
        """
        self.test_set_path = test_set_path
        self.test_set = {}
        self.results = []
        self.load_test_set()
        
    def load_test_set(self):
        """加载测试集"""
        try:
            with open(self.test_set_path, 'r', encoding='utf-8') as f:
                self.test_set = json.load(f)
            print(f"成功加载测试集，包含 {len(self.test_set['test_cases'])} 个测试用例")
        except Exception as e:
            print(f"加载测试集失败: {e}")
            raise
    
    def execute_test_case(self, test_case: Dict[str, Any], model_response: str) -> Dict[str, Any]:
        """
        执行单个测试用例
        
        Args:
            test_case: 测试用例
            model_response: 模型回答
            
        Returns:
            测试结果
        """
        result = {
            "test_id": test_case["test_id"],
            "layer": test_case["layer"],
            "category": test_case.get("category", "未知"),
            "instruction": test_case["instruction"],
            "expected_answer": test_case["expected_answer"],
            "model_response": model_response,
            "execution_time": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # 这里可以添加更复杂的评估逻辑
        # 目前使用简单的关键词匹配
        expected_keywords = test_case.get("expected_keywords", [])
        if expected_keywords:
            # 检查是否包含期望的关键词
            response_lower = model_response.lower()
            matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
            
            if matched_keywords:
                result["status"] = "passed"
                result["score"] = len(matched_keywords) / len(expected_keywords)
                result["matched_keywords"] = matched_keywords
            else:
                result["status"] = "failed"
                result["score"] = 0.0
                result["matched_keywords"] = []
        else:
            # 如果没有期望关键词，标记为需要人工评估
            result["status"] = "manual_review_needed"
            result["score"] = None
        
        return result
    
    def execute_all_tests(self, model_responses: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        执行所有测试用例
        
        Args:
            model_responses: 模型回答字典，key为test_id
            
        Returns:
            所有测试结果
        """
        print("开始执行综合测试...")
        
        results = []
        total_tests = len(self.test_set['test_cases'])
        
        for i, test_case in enumerate(self.test_set['test_cases']):
            test_id = test_case['test_id']
            
            if test_id in model_responses:
                result = self.execute_test_case(test_case, model_responses[test_id])
                results.append(result)
                
                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f"进度: {i + 1}/{total_tests}")
            else:
                print(f"警告: 测试用例 {test_id} 没有对应的模型回答")
                result = {
                    "test_id": test_id,
                    "layer": test_case["layer"],
                    "category": test_case.get("category", "未知"),
                    "instruction": test_case["instruction"],
                    "expected_answer": test_case["expected_answer"],
                    "model_response": "未提供",
                    "execution_time": datetime.now().isoformat(),
                    "status": "no_response",
                    "score": 0.0
                }
                results.append(result)
        
        self.results = results
        print(f"测试执行完成！共执行 {len(results)} 个测试用例")
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        if not self.results:
            print("没有测试结果，请先执行测试")
            return {}
        
        # 统计各层级结果
        layer_stats = {}
        for result in self.results:
            layer = result['layer']
            if layer not in layer_stats:
                layer_stats[layer] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'manual_review': 0,
                    'no_response': 0,
                    'scores': []
                }
            
            layer_stats[layer]['total'] += 1
            
            if result['status'] == 'passed':
                layer_stats[layer]['passed'] += 1
                if result['score'] is not None:
                    layer_stats[layer]['scores'].append(result['score'])
            elif result['status'] == 'failed':
                layer_stats[layer]['failed'] += 1
            elif result['status'] == 'manual_review_needed':
                layer_stats[layer]['manual_review'] += 1
            elif result['status'] == 'no_response':
                layer_stats[layer]['no_response'] += 1
        
        # 计算总体统计
        total_tests = len(self.results)
        total_passed = sum(1 for r in self.results if r['status'] == 'passed')
        total_failed = sum(1 for r in self.results if r['status'] == 'failed')
        total_manual = sum(1 for r in self.results if r['status'] == 'manual_review_needed')
        total_no_response = sum(1 for r in self.results if r['status'] == 'no_response')
        
        # 计算平均分数
        all_scores = [r['score'] for r in self.results if r['score'] is not None]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # 生成报告
        report = {
            "summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_manual_review": total_manual,
                "total_no_response": total_no_response,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
                "average_score": avg_score
            },
            "layer_statistics": layer_stats,
            "detailed_results": self.results,
            "generation_timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """保存测试报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_report_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"测试报告已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"保存测试报告失败: {e}")
            raise
    
    def export_results_to_excel(self, report: Dict[str, Any], output_path: str = None):
        """将测试结果导出到Excel"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_results_{timestamp}.xlsx"
        
        try:
            # 创建详细结果工作表
            detailed_df = pd.DataFrame(report['detailed_results'])
            
            # 创建统计摘要工作表
            summary_data = []
            for layer, stats in report['layer_statistics'].items():
                summary_data.append({
                    "测试层级": layer,
                    "总测试数": stats['total'],
                    "通过数": stats['passed'],
                    "失败数": stats['failed'],
                    "需人工审核": stats['manual_review'],
                    "无回答": stats['no_response'],
                    "通过率": f"{stats['passed']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%",
                    "平均分数": f"{sum(stats['scores'])/len(stats['scores']):.2f}" if stats['scores'] else "N/A"
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # 保存到Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                detailed_df.to_excel(writer, sheet_name='详细测试结果', index=False)
                summary_df.to_excel(writer, sheet_name='层级统计摘要', index=False)
            
            print(f"测试结果已导出到Excel: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"导出Excel失败: {e}")
            raise
    
    def print_summary(self, report: Dict[str, Any]):
        """打印测试摘要"""
        if not report:
            print("没有测试报告可显示")
            return
        
        summary = report['summary']
        print("\n" + "="*60)
        print("企业知识库缩略语模型测试报告")
        print("="*60)
        print(f"总测试用例数: {summary['total_tests']}")
        print(f"通过测试数: {summary['total_passed']}")
        print(f"失败测试数: {summary['total_failed']}")
        print(f"需人工审核: {summary['total_manual_review']}")
        print(f"无回答: {summary['total_no_response']}")
        print(f"总体通过率: {summary['pass_rate']*100:.1f}%")
        print(f"平均分数: {summary['average_score']:.2f}")
        
        print("\n各层级测试结果:")
        print("-" * 60)
        for layer, stats in report['layer_statistics'].items():
            pass_rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            print(f"{layer}: {stats['passed']}/{stats['total']} 通过 ({pass_rate:.1f}%), 平均分: {avg_score:.2f}")
        
        print("="*60)

def main():
    """主函数 - 演示用法"""
    print("=== 企业知识库测试执行器 ===")
    
    # 示例用法
    test_set_path = "comprehensive_test_set_20250101_120000.json"  # 替换为实际路径
    
    try:
        # 创建测试执行器
        executor = TestExecutor(test_set_path)
        
        # 模拟模型回答（实际使用时需要替换为真实的模型回答）
        print("\n注意: 这是演示模式，使用模拟数据")
        print("实际使用时，请提供真实的模型回答数据")
        
        # 模拟一些测试结果
        mock_results = executor.execute_all_tests({})
        
        # 生成报告
        report = executor.generate_report()
        
        # 保存报告
        json_path = executor.save_report(report)
        
        # 导出到Excel
        excel_path = executor.export_results_to_excel(report)
        
        # 打印摘要
        executor.print_summary(report)
        
        print(f"\n报告文件:")
        print(f"JSON格式: {json_path}")
        print(f"Excel格式: {excel_path}")
        
    except Exception as e:
        print(f"执行测试时发生错误: {e}")

if __name__ == "__main__":
    main()
