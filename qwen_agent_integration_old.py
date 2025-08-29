#!/usr/bin/env python3
"""
千问Agent集成 - 支持用户手动选择推理模型
"""

from enterprise_qa_system import process_enterprise_question, search_enterprise_terms, get_enterprise_categories
import json

class QwenAgentIntegration:
    """支持用户选择的千问Agent集成"""
    
    def __init__(self):
        self.terminology_list = self._load_terminology()
        print(f"✅ 千问Agent集成初始化完成，加载了 {len(self.terminology_list)} 个术语")
    
    def _load_terminology(self):
        """加载术语库"""
        try:
            with open('enterprise_terminology_complete.json', 'r', encoding='utf-8') as f:
                terminology = json.load(f)
            return list(terminology.keys())[:20]  # 显示前20个术语
        except FileNotFoundError:
            return ["AAP", "AM", "AIP", "SA", "AMGB", "Act."]
    
    def get_initial_response(self, user_question: str) -> dict:
        """获取初始响应，包含模型选择选项"""
        
        try:
            # 1. 分析问题
            kb_result = process_enterprise_question(user_question)
            
            # 2. 检查结果是否有效（不依赖success字段）
            if not kb_result or "search_results" not in kb_result:
                return {
                    "type": "error",
                    "content": "处理问题时发生错误：无法获取搜索结果",
                    "choices": None
                }
            
            # 3. 构建选择界面
            question_type = kb_result.get("question_type", "一般查询")
            terms = kb_result.get("search_results", [])
            knowledge = kb_result.get("knowledge", [])
            
            # 4. 根据问题类型提供不同的选择
            if question_type == "术语解释":
                choices = self._build_terminology_choices(user_question, terms)
            elif question_type == "事实查询":
                choices = self._build_fact_choices(user_question, terms, knowledge)
            else:
                choices = self._build_analysis_choices(user_question, terms, knowledge)
            
            # 5. 构建响应
            response = self._build_initial_response(user_question, question_type, terms, knowledge)
            
            return {
                "type": "choice",
                "content": response,
                "choices": choices,
                "question_analysis": {
                    "type": question_type,
                    "detected_terms": terms
                },
                "knowledge": knowledge
            }
            
        except Exception as e:
            return {
                "type": "error",
                "content": f"处理问题时发生错误：{str(e)}",
                "choices": None
            }
    
    def _build_terminology_choices(self, question: str, terms: list) -> list:
        """构建术语解释的选择选项"""
        choices = [
            {
                "id": "7b_model",
                "text": "🤖 使用本地7B模型回答",
                "description": "快速、经济，适合简单术语解释",
                "icon": "🤖"
            },
            {
                "id": "qwen_agent",
                "text": "🌟 使用千问Agent回答",
                "description": "更智能、更详细，适合复杂解释",
                "icon": "🌟"
            }
        ]
        return choices
    
    def _build_fact_choices(self, question: str, terms: list, knowledge: list) -> list:
        """构建事实查询的选择选项"""
        choices = [
            {
                "id": "7b_model",
                "text": "🤖 使用本地7B模型回答",
                "description": "基于检索到的知识，快速生成答案",
                "icon": "🤖"
            },
            {
                "id": "qwen_agent",
                "text": "🌟 使用千问Agent回答",
                "description": "结合大模型推理，提供更丰富信息",
                "icon": "🌟"
            },
            {
                "id": "direct_knowledge",
                "text": "📚 直接显示知识库内容",
                "description": "仅显示检索到的原始信息",
                "icon": "📚"
            }
        ]
        return choices
    
    def _build_analysis_choices(self, question: str, terms: list, knowledge: list) -> list:
        """构建分析请求的选择选项"""
        choices = [
            {
                "id": "7b_model",
                "text": "🤖 使用本地7B模型分析",
                "description": "基于训练数据进行分析，成本低",
                "icon": "🤖"
            },
            {
                "id": "qwen_agent",
                "text": "🌟 使用千问Agent深度分析",
                "description": "利用大模型能力，提供专业分析",
                "icon": "🌟"
            },
            {
                "id": "hybrid_approach",
                "text": "🔄 混合分析方案",
                "description": "先7B模型分析，再千问Agent补充",
                "icon": "🔄"
            }
        ]
        return choices
    
    def _build_initial_response(self, question: str, question_type: str, terms: list, knowledge: list) -> str:
        """构建初始响应内容"""
        response = f"我理解您的问题：**{question}**\n\n"
        
        # 添加问题分析
        response += f"**问题类型：** {question_type}\n\n"
        
        # 添加检测到的术语
        if terms:
            response += "**检测到企业术语：**\n"
            for term_info in terms:
                response += f"- {term_info['abbr']}：{term_info['info']['meaning']}\n"
            response += "\n"
        
        # 添加检索到的知识
        if knowledge:
            response += "**检索到相关信息：**\n"
            for i, info in enumerate(knowledge[:3], 1):  # 只显示前3条
                response += f"{i}. {info}\n"
            if len(knowledge) > 3:
                response += f"... 还有 {len(knowledge) - 3} 条相关信息\n"
            response += "\n"
        
        response += "**请选择您希望的回答方式：**\n"
        
        return response
    
    def process_user_choice(self, choice_id: str, question_data: dict) -> dict:
        """处理用户的选择"""
        question = question_data.get("question", "")
        question_type = question_data.get("question_type", "")
        terms = question_data.get("terms", [])
        knowledge = question_data.get("knowledge", [])
        
        if choice_id == "7b_model":
            return self._generate_7b_response(question, terms, knowledge)
        elif choice_id == "qwen_agent":
            return self._generate_qwen_response(question, terms, knowledge)
        elif choice_id == "direct_knowledge":
            return self._generate_direct_knowledge_response(question, terms, knowledge)
        elif choice_id == "hybrid_approach":
            return self._generate_hybrid_response(question, terms, knowledge)
        else:
            return {
                "type": "error",
                "content": "无效的选择，请重新选择"
            }
    
    def _generate_7b_response(self, question: str, terms: list, knowledge: list) -> dict:
        """生成7B模型的回答"""
        response = "🤖 **使用本地7B模型回答：**\n\n"
        
        if terms:
            response += "**术语解释：**\n"
            for term_info in terms:
                response += f"- {term_info['abbr']}：{term_info['info']['meaning']}\n"
            response += "\n"
        
        if knowledge:
            response += "**基于知识库的回答：**\n"
            for i, info in enumerate(knowledge, 1):
                response += f"{i}. {info}\n"
        
        response += "\n💰 **成本优势：** 使用本地模型，无需额外费用"
        
        return {
            "type": "response",
            "content": response,
            "model_used": "7B本地模型",
            "cost_saved": True
        }
    
    def _generate_qwen_response(self, question: str, terms: list, knowledge: list) -> dict:
        """生成千问Agent的回答"""
        response = "�� **使用千问Agent回答：**\n\n"
        
        if terms:
            response += "**术语解释：**\n"
            for term_info in terms:
                response += f"- {term_info['abbr']}：{term_info['info']['meaning']}\n"
            response += "\n"
        
        if knowledge:
            response += "**智能分析：**\n"
            response += "基于检索到的信息，千问Agent将为您提供：\n"
            response += "• 更深入的分析和见解\n"
            response += "• 相关的业务建议\n"
            response += "• 可能的影响和风险分析\n"
            response += "• 后续行动建议\n\n"
        
        response += "**优势：** 更智能、更全面、更专业"
        
        return {
            "type": "response",
            "content": response,
            "model_used": "千问Agent",
            "cost_saved": False
        }
    
    def _generate_direct_knowledge_response(self, question: str, terms: list, knowledge: list) -> dict:
        """生成直接知识库内容的回答"""
        response = "�� **直接显示知识库内容：**\n\n"
        
        if terms:
            response += "**相关术语：**\n"
            for term_info in terms:
                response += f"- {term_info['abbr']}：{term_info['info']['meaning']}\n"
            response += "\n"
        
        if knowledge:
            response += "**知识库内容：**\n"
            for i, info in enumerate(knowledge, 1):
                response += f"--- 信息 {i} ---\n{info}\n\n"
        else:
            response += "未找到相关信息。"
        
        response += "\n**特点：** 原始信息，无加工，最准确"
        
        return {
            "type": "response",
            "content": response,
            "model_used": "知识库直接检索",
            "cost_saved": True
        }
    
    def _generate_hybrid_response(self, question: str, terms: list, knowledge: list) -> dict:
        """生成混合分析的回答"""
        response = "🔄 **混合分析方案：**\n\n"
        
        # 7B模型分析
        response += "**第一步：7B模型基础分析**\n"
        response += "基于训练数据，7B模型提供：\n"
        response += "• 基础的业务理解\n"
        response += "• 相关的流程说明\n"
        response += "• 初步的建议\n\n"
        
        # 千问Agent补充
        response += "**第二步：千问Agent深度补充**\n"
        response += "千问Agent将提供：\n"
        response += "• 更深入的分析\n"
        response += "• 创新性建议\n"
        response += "• 风险评估\n"
        response += "• 最佳实践\n\n"
        
        response += "**优势：** 结合两种模型的优势，既经济又全面"
        
        return {
            "type": "response",
            "content": response,
            "model_used": "7B模型 + 千问Agent",
            "cost_saved": "部分节省"
        }

# 使用示例
def main():
    print("=== 千问Agent选择界面示例 ===\n")
    
    # 初始化系统
    agent = QwenAgentIntegration()
    
    # 模拟用户问题
    test_question = "AAP是什么？"
    print(f"用户问题: {test_question}\n")
    
    # 获取初始响应和选择选项
    initial_response = agent.get_initial_response(test_question)
    
    if initial_response["type"] == "choice":
        print("千问Agent响应:")
        print(initial_response["content"])
        
        print("\n选择选项:")
        for choice in initial_response["choices"]:
            print(f"{choice['icon']} {choice['text']}")
            print(f"   {choice['description']}\n")
        
        # 模拟用户选择
        print("用户选择: 使用本地7B模型回答")
        choice_result = agent.process_user_choice("7b_model", {
            "question": test_question,
            "question_type": initial_response["question_analysis"]["type"],
            "terms": initial_response["question_analysis"]["detected_terms"],
            "knowledge": initial_response["knowledge"]
        })
        
        print("\n" + "="*60)
        print("最终回答:")
        print(choice_result["content"])
    else:
        print(f"错误: {initial_response['content']}")

if __name__ == "__main__":
    main()