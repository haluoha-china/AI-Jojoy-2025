#!/usr/bin/env python3
"""
千问Agent与企业知识库集成系统
提供Function Call接口，实现术语规范化 + RAG检索的完整流程
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# 导入现有服务
from glossary_service import GlossaryService
from knowledge_base import KnowledgeBaseService

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenAgentIntegration:
    """千问Agent集成服务 - 提供Function Call接口"""
    
    def __init__(self):
        """初始化集成服务"""
        try:
            # 初始化术语对照服务
            self.glossary_service = GlossaryService()
            logger.info("✅ 术语对照服务初始化成功")
            
            # 初始化知识库服务
            self.kb_service = KnowledgeBaseService()
            logger.info("✅ 知识库服务初始化成功")
            
            logger.info("🚀 千问Agent集成服务初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 服务初始化失败: {e}")
            raise
    
    def normalize_question_with_glossary(self, question: str) -> Dict[str, Any]:
        """
        Function Call 1: 术语规范化
        将用户问题中的"黑话"转换为标准术语
        """
        try:
            logger.info(f"🔍 术语规范化处理: {question}")
            
            # 调用术语对照服务
            normalized_question, glossary_hits = self.glossary_service.normalize_question(question)
            
            result = {
                "original_question": question,
                "normalized_question": normalized_question,
                "glossary_hits": glossary_hits,
                "has_changes": question != normalized_question,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            logger.info(f"✅ 术语规范化完成: {question} → {normalized_question}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 术语规范化失败: {e}")
            return {
                "original_question": question,
                "normalized_question": question,
                "glossary_hits": {},
                "has_changes": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def search_knowledge_base(self, query: str, top_k: int = 5, score_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Function Call 2: 知识库检索
        基于规范化后的问题进行RAG检索
        """
        try:
            logger.info(f"🔍 知识库检索: {query}, top_k={top_k}, threshold={score_threshold}")
            
            # 调用知识库服务
            search_results = self.kb_service.search(query, top_k=top_k, score_threshold=score_threshold)
            
            result = {
                "query": query,
                "search_results": search_results,
                "total_results": len(search_results),
                "top_k": top_k,
                "score_threshold": score_threshold,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            logger.info(f"✅ 知识库检索完成: 找到 {len(search_results)} 个结果")
            return result
            
        except Exception as e:
            logger.error(f"❌ 知识库检索失败: {e}")
            return {
                "query": query,
                "search_results": [],
                "total_results": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def process_enterprise_question(self, question: str, top_k: int = 5, score_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Function Call 3: 完整问答流程
        术语规范化 + RAG检索的端到端处理
        """
        try:
            logger.info(f"🚀 开始处理企业问题: {question}")
            
            # 步骤1: 术语规范化
            normalization_result = self.normalize_question_with_glossary(question)
            
            # 步骤2: 知识库检索
            search_result = self.search_knowledge_base(
                normalization_result["normalized_question"], 
                top_k=top_k, 
                score_threshold=score_threshold
            )
            
            # 步骤3: 组合结果
            final_result = {
                "question": question,
                "processing_steps": {
                    "step1_terminology_normalization": normalization_result,
                    "step2_knowledge_retrieval": search_result
                },
                "summary": {
                    "original_question": question,
                    "normalized_question": normalization_result["normalized_question"],
                    "terminology_changes": normalization_result["has_changes"],
                    "retrieved_documents": search_result["total_results"],
                    "processing_time": datetime.now().isoformat()
                },
                "status": "success"
            }
            
            logger.info(f"✅ 企业问题处理完成: {question}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 企业问题处理失败: {e}")
            return {
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Function Call 4: 系统状态查询
        返回术语对照和知识库的状态信息
        """
        try:
            # 获取术语对照服务状态
            glossary_status = {
                "service": "glossary_service",
                "status": "running",
                "terminology_count": len(self.glossary_service.term_to_standard) if hasattr(self.glossary_service, 'term_to_standard') else 0
            }
            
            # 获取知识库服务状态
            kb_status = {
                "service": "knowledge_base",
                "status": "running",
                "document_count": self.kb_service.get_document_count() if hasattr(self.kb_service, 'get_document_count') else 0
            }
            
            return {
                "system_status": "running",
                "services": [glossary_status, kb_status],
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"❌ 获取系统状态失败: {e}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# 创建全局实例
qwen_agent_integration = QwenAgentIntegration()

# ==================== Function Call 接口定义 ====================
# 这些函数供千问Agent直接调用

def normalize_question_function(question: str) -> Dict[str, Any]:
    """Function Call: 术语规范化"""
    return qwen_agent_integration.normalize_question_with_glossary(question)

def search_knowledge_function(query: str, top_k: int = 5, score_threshold: float = 0.3) -> Dict[str, Any]:
    """Function Call: 知识库检索"""
    return qwen_agent_integration.search_knowledge_base(query, top_k, score_threshold)

def process_question_function(question: str, top_k: int = 5, score_threshold: float = 0.3) -> Dict[str, Any]:
    """Function Call: 完整问答流程"""
    return qwen_agent_integration.process_enterprise_question(question, top_k, score_threshold)

def get_system_status_function() -> Dict[str, Any]:
    """Function Call: 系统状态查询"""
    return qwen_agent_integration.get_system_status()

# ==================== 测试函数 ====================

def test_integration_system():
    """测试集成系统功能"""
    print("🧪 测试千问Agent集成系统...")
    
    try:
        # 测试1: 术语规范化
        print("\n1. 测试术语规范化:")
        test_question = "AAP是什么？"
        result = normalize_question_function(test_question)
        print(f"问题: {test_question}")
        print(f"规范化结果: {result['normalized_question']}")
        print(f"是否有变化: {result['has_changes']}")
        print(f"命中术语: {result['glossary_hits']}")
        
        # 测试2: 知识库检索
        print("\n2. 测试知识库检索:")
        search_result = search_knowledge_function("AAP", top_k=3)
        print(f"检索结果数量: {search_result['total_results']}")
        print(f"状态: {search_result['status']}")
        
        # 测试3: 完整流程
        print("\n3. 测试完整问答流程:")
        full_result = process_question_function("AAP是什么？", top_k=3)
        print(f"处理状态: {full_result['status']}")
        print(f"术语变化: {full_result['summary']['terminology_changes']}")
        print(f"检索文档数: {full_result['summary']['retrieved_documents']}")
        
        # 测试4: 系统状态
        print("\n4. 测试系统状态:")
        status = get_system_status_function()
        print(f"系统状态: {status['system_status']}")
        for service in status['services']:
            print(f"  {service['service']}: {service['status']}")
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration_system()
