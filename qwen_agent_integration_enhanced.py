#!/usr/bin/env python3
"""
千问Agent集成（增强版）
- 融合新版的Function Call接口与旧版的模型选择能力
- 流程：术语规范化 → RAG检索 → 生成策略（7B/千问/混合）
- 生成阶段暂为占位（不直接实现调用），先完成前两步联调
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from glossary_service import GlossaryService
from knowledge_base import KnowledgeBaseService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenAgentIntegrationEnhanced:
    """融合版：提供Function Call与用户策略选择入口"""

    def __init__(self):
        self.glossary = GlossaryService()
        self.kb = KnowledgeBaseService()
        logger.info("✅ 增强版集成初始化完成")

    # ---------- Function Calls ----------
    def fc_normalize(self, question: str) -> Dict[str, Any]:
        """术语规范化（Function Call）"""
        normalized, hits = self.glossary.normalize_question(question)
        return {
            "original_question": question,
            "normalized_question": normalized,
            "glossary_hits": hits,
            "has_changes": normalized != question,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    def fc_retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """知识库检索（Function Call）"""
        results = self.kb.search(query, top_k=top_k)
        return {
            "query": query,
            "search_results": results,
            "total_results": len(results),
            "top_k": top_k,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    def fc_process(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """端到端流程（规范化+检索）"""
        norm = self.fc_normalize(question)
        retr = self.fc_retrieve(norm["normalized_question"], top_k=top_k)
        return {
            "question": question,
            "steps": {
                "normalization": norm,
                "retrieval": retr
            },
            "summary": {
                "normalized_question": norm["normalized_question"],
                "terminology_changes": norm["has_changes"],
                "retrieved_documents": retr["total_results"]
            },
            "status": "success"
        }

    # ---------- 用户策略选择（借鉴旧版） ----------
    def build_choices(self, question_type: str) -> List[Dict[str, str]]:
        choices = [
            {"id": "local_7b", "text": "🤖 使用本地7B模型", "description": "经济快捷，适合术语/事实类", "icon": "🤖"},
            {"id": "qwen", "text": "🌟 使用千问Agent", "description": "高质量生成，适合复杂解释/分析", "icon": "🌟"},
            {"id": "hybrid", "text": "🔄 混合策略", "description": "先7B后千问，兼顾成本与质量", "icon": "🔄"}
        ]
        return choices

    def initial_response(self, question: str) -> Dict[str, Any]:
        """构建带有策略选择的初始响应（不做生成，仅展示选择）"""
        proc = self.fc_process(question, top_k=3)
        question_type = "术语解释" if proc["summary"]["retrieved_documents"] > 0 else "一般查询"
        choices = self.build_choices(question_type)
        content = f"我理解您的问题：**{question}**\n\n" \
                  f"**检索到文档数：** {proc['summary']['retrieved_documents']}\n\n" \
                  f"请选择您希望的回答方式：\n"
        return {
            "type": "choice",
            "content": content,
            "choices": choices,
            "analysis": {
                "type": question_type,
                "normalized_question": proc["summary"]["normalized_question"],
                "terminology_changes": proc["summary"]["terminology_changes"],
                "retrieved_documents": proc["summary"]["retrieved_documents"],
            },
            "steps": proc["steps"],
            "timestamp": datetime.now().isoformat()
        }

# --------- Function Call 入口（供千问Agent调用） ---------
_enhanced = QwenAgentIntegrationEnhanced()

def normalize_question_function(question: str) -> Dict[str, Any]:
    return _enhanced.fc_normalize(question)

def search_knowledge_function(query: str, top_k: int = 5) -> Dict[str, Any]:
    return _enhanced.fc_retrieve(query, top_k=top_k)

def process_question_function(question: str, top_k: int = 5) -> Dict[str, Any]:
    return _enhanced.fc_process(question, top_k=top_k)

def initial_response_function(question: str) -> Dict[str, Any]:
    return _enhanced.initial_response(question)

# --------- 本地快速测试 ---------
if __name__ == "__main__":
    print("🧪 增强版联调测试\n")
    q = "AAP是什么？"
    print("1) 端到端处理:")
    print(process_question_function(q))
    print("\n2) 初始响应（带选择）:")
    print(initial_response_function(q))
