#!/usr/bin/env python3
"""
企业问答系统 - 供千问Agent调用
强调：不要幻觉，必须基于知识库检索结果回答
"""

import json
import re
import os
from typing import Dict, List, Tuple

class EnterpriseQASystem:
    """企业问答系统 - 严格基于知识库，禁止幻觉"""
    
    def __init__(self):
        self.terminology = self._load_terminology()
        self.categories = self._load_categories()
        print(f"✅ 企业问答系统初始化完成，加载了 {len(self.terminology)} 个术语")
        print("⚠️ 重要提醒：系统严格基于知识库回答，禁止生成幻觉信息")
    
    def _load_terminology(self):
        """加载术语库"""
        try:
            with open('enterprise_terminology_complete.json', 'r', encoding='utf-8') as f:
                terminology = json.load(f)
            return terminology
        except FileNotFoundError:
            print("⚠️ 术语库文件未找到，请先运行 build_complete_terminology.py")
            return {}
    
    def _load_categories(self):
        """加载分类信息"""
        try:
            with open('enterprise_abbreviations_categories.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ 分类文件未找到，请先运行 build_complete_terminology.py")
            return {}
    
    def search_terminology(self, query: str) -> List[Dict]:
        """搜索术语 - 严格基于知识库"""
        results = []
        query_lower = query.lower()
        
        for abbr, info in self.terminology.items():
            # 搜索缩略语（精确匹配）
            if query_lower == abbr.lower():
                results.append({
                    "abbr": abbr,
                    "info": info,
                    "match_type": "缩略语精确匹配",
                    "score": 1.0,
                    "source": info.get('source', 'unknown'),
                    "line": info.get('line', 'unknown')
                })
                continue
            
            # 搜索缩略语（包含匹配）
            if query_lower in abbr.lower():
                results.append({
                    "abbr": abbr,
                    "info": info,
                    "match_type": "缩略语包含匹配",
                    "score": 0.9,
                    "source": info.get('source', 'unknown'),
                    "line": info.get('line', 'unknown')
                })
                continue
            
            # 搜索英文全称
            if query_lower in info['full'].lower():
                results.append({
                    "abbr": abbr,
                    "info": info,
                    "match_type": "英文全称匹配",
                    "score": 0.8,
                    "source": info.get('source', 'unknown'),
                    "line": info.get('line', 'unknown')
                })
                continue
            
            # 搜索中文含义
            if query_lower in info['meaning'].lower():
                results.append({
                    "abbr": abbr,
                    "info": info,
                    "match_type": "中文含义匹配",
                    "score": 0.8,
                    "source": info.get('source', 'unknown'),
                    "line": info.get('line', 'unknown')
                })
                continue
        
        # 按匹配分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_category_info(self, category_name: str) -> Dict:
        """获取分类信息"""
        if category_name in self.categories:
            return {
                "category": category_name,
                "count": len(self.categories[category_name]),
                "terms": self.categories[category_name]
            }
        return None
    
    def get_all_categories(self) -> Dict:
        """获取所有分类信息"""
        return {
            category: {
                "count": len(terms),
                "sample_terms": terms[:5]  # 显示前5个术语作为示例
            }
            for category, terms in self.categories.items()
            if terms
        }
    
    def process_question(self, question: str) -> Dict:
        """处理用户问题 - 严格基于知识库，禁止幻觉"""
        # 1. 搜索相关术语
        search_results = self.search_terminology(question)
        
        # 2. 分析问题类型
        question_type = self._analyze_question_type(question)
        
        # 3. 构建回答 - 必须包含来源信息
        response = self._build_response(question, search_results, question_type)
        
        return {
            "question": question,
            "question_type": question_type,
            "search_results": search_results,
            "response": response,
            "categories_info": self.get_all_categories(),
            "disclaimer": "⚠️ 重要提醒：所有回答均基于企业知识库检索结果，禁止生成幻觉信息"
        }
    
    def _analyze_question_type(self, question: str) -> str:
        """分析问题类型"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["是什么", "什么意思", "含义", "解释", "代表什么", "缩写"]):
            return "术语解释"
        elif any(keyword in question_lower for keyword in ["哪个", "谁", "哪一位", "什么时候", "何时", "时间"]):
            return "事实查询"
        elif any(keyword in question_lower for keyword in ["如何", "怎样", "步骤", "建议", "分析"]):
            return "分析请求"
        else:
            return "一般查询"
    
    def _build_response(self, question: str, search_results: List, question_type: str) -> str:
        """构建回答 - 必须包含来源信息，禁止幻觉"""
        if not search_results:
            return "⚠️ **重要提醒：基于企业知识库检索结果**\n\n抱歉，我在企业术语库中没有找到相关信息。\n\n**系统原则：**\n- 严格基于知识库回答\n- 禁止生成幻觉信息\n- 如果知识库中没有相关信息，请明确说明\n\n请尝试使用其他关键词搜索，或联系相关部门获取最新信息。"
        
        response_parts = []
        
        # 添加重要提醒
        response_parts.append("⚠️ **重要提醒：以下回答严格基于企业知识库检索结果，禁止生成幻觉信息**\n")
        
        # 添加问题分析
        response_parts.append(f"**问题分析：** {question_type}")
        response_parts.append(f"**知识库检索结果：** 找到 {len(search_results)} 个相关术语\n")
        
        # 添加搜索结果 - 必须包含来源信息
        for i, result in enumerate(search_results[:10], 1):  # 最多显示10个结果
            abbr = result['abbr']
            info = result['info']
            match_type = result['match_type']
            source = result.get('source', 'unknown')
            line = result.get('line', 'unknown')
            
            response_parts.append(f"{i}. **{abbr}**")
            response_parts.append(f"   - 英文：{info['full']}")
            response_parts.append(f"   - 中文：{info['meaning']}")
            response_parts.append(f"   - 分类：{info['category']}")
            response_parts.append(f"   - 匹配：{match_type}")
            response_parts.append(f"   - 数据来源：{source}")
            response_parts.append(f"   - 行号：{line}")
            response_parts.append("")
        
        if len(search_results) > 10:
            response_parts.append(f"... 还有 {len(search_results) - 10} 个结果")
        
        # 添加分类信息
        response_parts.append("**企业术语库分类概览：**")
        categories_info = self.get_all_categories()
        for category, info in categories_info.items():
            response_parts.append(f"- {category}: {info['count']} 个术语")
        
        # 添加系统原则
        response_parts.append("\n**系统回答原则：**")
        response_parts.append("1. ✅ 严格基于知识库检索结果")
        response_parts.append("2. ✅ 必须包含数据来源和行号信息")
        response_parts.append("3. ❌ 禁止生成幻觉信息")
        response_parts.append("4. ❌ 禁止添加知识库中没有的内容")
        response_parts.append("5. ❌ 禁止推测或假设")
        
        return "\n".join(response_parts)
    


    def _extract_keywords(self, question: str) -> List[str]:
        """适度提取关键词 - 简单但有效"""
        keywords = []
        
        # 1. 移除常见问题词和连接词
        question_clean = question
        question_words = ["是什么", "什么意思", "含义", "解释", "代表什么", "缩写", "？", "?", "请", "我想了解", "告诉我", "的", "什么"]
        for word in question_words:
            question_clean = question_clean.replace(word, "")
        
        # 2. 提取可能的缩略语（大写字母组合）
        abbr_pattern = r'\b[A-Z]{2,}\b'
        abbrs = re.findall(abbr_pattern, question_clean)
        keywords.extend(abbrs)
        
        # 3. 提取其他关键词（过滤掉太短的词）
        words = question_clean.strip().split()
        keywords.extend([word for word in words if len(word) > 1 and word not in ["的", "什么", "是", "有", "在", "和", "与"]])
        
        # 4. 去重并返回
        return list(set(keywords))
    def search_terminology_smart(self, question: str) -> List[Dict]:
        """智能搜索 - 适度理解问题"""
        # 1. 提取关键词
        keywords = self._extract_keywords(question)
        print(f"提取的关键词: {keywords}")
        
        # 2. 尝试每个关键词
        all_results = []
        for keyword in keywords:
            results = self.search_terminology(keyword)
            all_results.extend(results)
        
        # 3. 去重并按分数排序
        seen = set()
        unique_results = []
        for result in all_results:
            if result['abbr'] not in seen:
                seen.add(result['abbr'])
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results
    def get_source_info(self, abbr: str) -> Dict:
        """获取特定术语的详细来源信息"""
        if abbr in self.terminology:
            info = self.terminology[abbr]
            return {
                "abbr": abbr,
                "full": info['full'],
                "meaning": info['meaning'],
                "category": info['category'],
                "source": info.get('source', 'unknown'),
                "line": info.get('line', 'unknown'),
                "instruction": info.get('instruction', ''),
                "output": info.get('output', '')
            }
        return None

# 创建全局实例
enterprise_qa_system = EnterpriseQASystem()

# 主要接口函数 - 供千问Agent调用
def search_enterprise_terms(query: str) -> List[Dict]:
    """搜索企业术语 - 严格基于知识库"""
    return enterprise_qa_system.search_terminology(query)

def get_enterprise_categories() -> Dict:
    """获取企业术语分类"""
    return enterprise_qa_system.get_all_categories()

def process_enterprise_question(question: str) -> Dict:
    """处理企业相关问题 - 严格基于知识库，禁止幻觉"""
    return enterprise_qa_system.process_question(question)

def get_term_source_info(abbr: str) -> Dict:
    """获取术语的详细来源信息"""
    return enterprise_qa_system.get_source_info(abbr)

# 测试函数
def test_system():
    """测试系统功能"""
    print("=== 企业问答系统测试 - 严格基于知识库版本 ===\n")
    
    # 测试搜索
    print("1. 测试术语搜索:")
    results = search_enterprise_terms("打印")
    print(f"搜索'打印'找到 {len(results)} 个结果")
    if results:
        first_result = results[0]
        print(f"第一个结果: {first_result['abbr']} - {first_result['info']['meaning']}")
        print(f"来源: {first_result['source']}, 行号: {first_result['line']}")
    
    print("\n2. 测试分类信息:")
    categories = get_enterprise_categories()
    print(f"共有 {len(categories)} 个分类")
    for category, info in list(categories.items())[:3]:
        print(f"  {category}: {info['count']} 个术语")
    
    print("\n3. 测试问题处理:")
    test_question = "AAP是什么？"
    result = process_enterprise_question(test_question)
    print(f"问题: {test_question}")
    print(f"问题类型: {result['question_type']}")
    print(f"找到术语: {len(result['search_results'])} 个")
    print(f"免责声明: {result['disclaimer']}")

if __name__ == "__main__":
    test_system()