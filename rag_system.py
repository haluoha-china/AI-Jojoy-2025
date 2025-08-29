"""
企业知识库 - RAG问答系统
实现检索增强生成（Retrieval-Augmented Generation）
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from vectorizer import Vectorizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG问答系统主类"""
    
    def __init__(self, vectorizer: Vectorizer):
        """
        初始化RAG系统
        
        Args:
            vectorizer: 向量化器实例
        """
        self.vectorizer = vectorizer
        self.conversation_history = []
        self.max_history = 10
        
        # 系统提示词模板
        self.system_prompt = """你是一个专业的企业知识库助手，能够基于检索到的文档内容回答用户问题。
请遵循以下原则：
1. 只基于检索到的文档内容回答问题
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、专业
4. 引用具体的文档来源（页码、章节等）
5. 如果信息不完整，请说明需要补充的内容"""
    
    def answer_question(self, question: str, 
                       top_k: int = 5, 
                       score_threshold: float = 0.3,
                       include_context: bool = True) -> Dict[str, Any]:
        """
        回答问题
        
        Args:
            question: 用户问题
            top_k: 检索的文档块数量
            score_threshold: 相似度阈值
            include_context: 是否包含上下文
            
        Returns:
            答案和相关信息
        """
        try:
            logger.info(f"开始处理问题: {question}")
            
            # 记录问题时间
            question_time = datetime.now()
            
            # 检索相关文档
            relevant_chunks = self.vectorizer.search(
                question, 
                top_k=top_k, 
                score_threshold=score_threshold
            )
            
            if not relevant_chunks:
                return self._generate_no_answer_response(question, question_time)
            
            # 构建上下文
            context = self._build_context(relevant_chunks, include_context)
            
            # 生成答案
            answer = self._generate_answer(question, context, relevant_chunks)
            
            # 构建响应
            response = {
                'question': question,
                'answer': answer,
                'context': context if include_context else None,
                'sources': self._format_sources(relevant_chunks),
                'metadata': {
                    'question_time': question_time.isoformat(),
                    'answer_time': datetime.now().isoformat(),
                    'chunks_retrieved': len(relevant_chunks),
                    'top_similarity': relevant_chunks[0]['similarity'] if relevant_chunks else 0,
                    'score_threshold': score_threshold
                }
            }
            
            # 添加到对话历史
            self._add_to_history(question, response)
            
            logger.info(f"问题处理完成，检索到 {len(relevant_chunks)} 个相关文档块")
            return response
            
        except Exception as e:
            logger.error(f"问题处理失败: {e}")
            return self._generate_error_response(question, str(e))
    
    def _generate_no_answer_response(self, question: str, question_time: datetime) -> Dict[str, Any]:
        """生成无答案响应"""
        return {
            'question': question,
            'answer': "抱歉，我在知识库中没有找到与您问题相关的信息。请尝试：\n1. 使用不同的关键词重新提问\n2. 检查问题是否与已上传的文档内容相关\n3. 联系管理员添加相关文档",
            'context': None,
            'sources': [],
            'metadata': {
                'question_time': question_time.isoformat(),
                'answer_time': datetime.now().isoformat(),
                'chunks_retrieved': 0,
                'top_similarity': 0,
                'score_threshold': 0,
                'status': 'no_relevant_content'
            }
        }
    
    def _generate_error_response(self, question: str, error_msg: str) -> Dict[str, Any]:
        """生成错误响应"""
        return {
            'question': question,
            'answer': f"抱歉，处理您的问题时出现了错误：{error_msg}。请稍后重试或联系技术支持。",
            'context': None,
            'sources': [],
            'metadata': {
                'question_time': datetime.now().isoformat(),
                'answer_time': datetime.now().isoformat(),
                'chunks_retrieved': 0,
                'top_similarity': 0,
                'score_threshold': 0,
                'status': 'error',
                'error_message': error_msg
            }
        }
    
    def _build_context(self, relevant_chunks: List[Dict], include_context: bool) -> str:
        """
        构建上下文
        
        Args:
            relevant_chunks: 相关文档块
            include_context: 是否包含上下文
            
        Returns:
            上下文字符串
        """
        if not include_context:
            return ""
        
        context_parts = []
        
        for chunk_info in relevant_chunks:
            chunk = chunk_info['chunk']
            similarity = chunk_info['similarity']
            
            # 构建上下文段落
            context_part = f"【第{chunk.get('page_num', '未知')}页】"
            if chunk.get('chunk_type'):
                context_part += f" ({chunk['chunk_type']})"
            context_part += f" 相关度: {similarity:.3f}\n"
            context_part += chunk['content']
            context_part += "\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 上下文信息
            sources: 来源文档块
            
        Returns:
            生成的答案
        """
        # 这里可以集成大语言模型来生成更智能的答案
        # 目前使用模板生成
        
        if not sources:
            return "抱歉，没有找到相关信息。"
        
        # 分析问题类型
        question_type = self._classify_question(question)
        
        # 根据问题类型生成答案
        if question_type == "definition":
            return self._generate_definition_answer(question, context, sources)
        elif question_type == "comparison":
            return self._generate_comparison_answer(question, context, sources)
        elif question_type == "procedure":
            return self._generate_procedure_answer(question, context, sources)
        elif question_type == "data":
            return self._generate_data_answer(question, context, sources)
        else:
            return self._generate_general_answer(question, context, sources)
    
    def _classify_question(self, question: str) -> str:
        """分类问题类型"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['是什么', '定义', '概念', '含义']):
            return "definition"
        elif any(word in question_lower for word in ['比较', '区别', '差异', 'vs', '对比']):
            return "comparison"
        elif any(word in question_lower for word in ['如何', '怎么', '步骤', '流程', '方法']):
            return "procedure"
        elif any(word in question_lower for word in ['数据', '数字', '统计', '金额', '数量']):
            return "data"
        else:
            return "general"
    
    def _generate_definition_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """生成定义类问题的答案"""
        answer = "根据检索到的文档内容，"
        
        # 提取最相关的定义内容
        top_source = sources[0]
        chunk = top_source['chunk']
        
        # 尝试找到定义句
        content = chunk['content']
        sentences = content.split('。')
        
        for sentence in sentences:
            if any(word in sentence for word in ['是', '指', '定义', '称为']):
                answer += sentence.strip() + "。"
                break
        else:
            # 如果没有找到明确的定义句，使用整个内容
            answer += content[:200] + "..." if len(content) > 200 else content
        
        answer += f"\n\n信息来源：第{chunk.get('page_num', '未知')}页"
        return answer
    
    def _generate_comparison_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """生成比较类问题的答案"""
        answer = "根据检索到的文档内容，"
        
        # 分析多个来源，找出比较信息
        comparison_points = []
        
        for source in sources[:3]:  # 最多分析3个来源
            chunk = source['chunk']
            content = chunk['content']
            
            # 查找比较关键词
            if any(word in content for word in ['而', '但是', '然而', '相比', '不同']):
                comparison_points.append(f"第{chunk.get('page_num', '未知')}页：{content[:100]}...")
        
        if comparison_points:
            answer += "相关信息如下：\n" + "\n".join(comparison_points)
        else:
            answer += "文档中包含了相关的内容，但可能需要更详细的比较分析。"
        
        answer += f"\n\n建议查看完整文档内容以获得更全面的比较信息。"
        return answer
    
    def _generate_procedure_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """生成流程类问题的答案"""
        answer = "根据检索到的文档内容，"
        
        # 查找流程相关信息
        procedure_steps = []
        
        for source in sources:
            chunk = source['chunk']
            content = chunk['content']
            
            # 查找步骤标识
            if any(word in content for word in ['第一步', '首先', '然后', '接着', '最后']):
                procedure_steps.append(f"第{chunk.get('page_num', '未知')}页：{content}")
        
        if procedure_steps:
            answer += "相关流程如下：\n" + "\n".join(procedure_steps)
        else:
            answer += "文档中包含了相关的内容，建议查看完整文档以了解详细流程。"
        
        return answer
    
    def _generate_data_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """生成数据类问题的答案"""
        answer = "根据检索到的文档内容，"
        
        # 查找数据信息
        data_info = []
        
        for source in sources:
            chunk = source['chunk']
            content = chunk['content']
            
            # 查找数字信息
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            if numbers:
                data_info.append(f"第{chunk.get('page_num', '未知')}页：{content}")
        
        if data_info:
            answer += "相关数据信息如下：\n" + "\n".join(data_info[:3])  # 最多显示3个
        else:
            answer += "文档中包含了相关的内容，但可能需要查看表格或图表获取具体数据。"
        
        return answer
    
    def _generate_general_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """生成一般问题的答案"""
        answer = "根据检索到的文档内容，"
        
        # 使用最相关的来源
        top_source = sources[0]
        chunk = top_source['chunk']
        
        answer += chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content']
        
        # 如果有多个来源，简要提及
        if len(sources) > 1:
            answer += f"\n\n此外，在第{', 第'.join(str(s['chunk'].get('page_num', '未知')) for s in sources[1:3])}页也有相关信息。"
        
        answer += f"\n\n信息来源：第{chunk.get('page_num', '未知')}页"
        return answer
    
    def _format_sources(self, sources: List[Dict]) -> List[Dict]:
        """格式化来源信息"""
        formatted_sources = []
        
        for source in sources:
            chunk = source['chunk']
            formatted_source = {
                'chunk_id': chunk.get('chunk_id', ''),
                'page_num': chunk.get('page_num', '未知'),
                'chunk_type': chunk.get('chunk_type', ''),
                'similarity': source['similarity'],
                'content_preview': chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content'],
                'metadata': chunk.get('metadata', {})
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def _add_to_history(self, question: str, response: Dict):
        """添加到对话历史"""
        history_item = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response
        }
        
        self.conversation_history.append(history_item)
        
        # 保持历史记录数量限制
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """获取对话历史"""
        if limit is None:
            return self.conversation_history.copy()
        else:
            return self.conversation_history[-limit:]
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("对话历史已清空")
    
    def batch_answer_questions(self, questions: List[str], 
                             top_k: int = 5, 
                             score_threshold: float = 0.3) -> List[Dict]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            top_k: 检索的文档块数量
            score_threshold: 相似度阈值
            
        Returns:
            答案列表
        """
        answers = []
        
        for question in questions:
            try:
                answer = self.answer_question(
                    question, 
                    top_k=top_k, 
                    score_threshold=score_threshold
                )
                answers.append(answer)
            except Exception as e:
                logger.error(f"处理问题失败: {question}, 错误: {e}")
                error_response = self._generate_error_response(question, str(e))
                answers.append(error_response)
        
        return answers
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            'total_conversations': len(self.conversation_history),
            'vectorizer_info': self.vectorizer.get_index_info(),
            'system_prompt': self.system_prompt,
            'max_history': self.max_history
        }
        
        return stats
    
    def update_system_prompt(self, new_prompt: str):
        """更新系统提示词"""
        self.system_prompt = new_prompt
        logger.info("系统提示词已更新")
    
    def export_conversation_history(self, filepath: str, format: str = 'json'):
        """导出对话历史"""
        try:
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"对话历史已导出到: {filepath}")
            
        except Exception as e:
            logger.error(f"导出对话历史失败: {e}")
            raise

# 使用示例
if __name__ == "__main__":
    # 创建向量化器（需要先有文本块数据）
    from vectorizer import Vectorizer
    
    # 示例文本块
    sample_chunks = [
        {
            'chunk_id': 'chunk_1',
            'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
            'page_num': 1,
            'chunk_type': 'sentence'
        },
        {
            'chunk_id': 'chunk_2',
            'content': '机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。',
            'page_num': 1,
            'chunk_type': 'sentence'
        }
    ]
    
    # 创建向量化器并构建索引
    vectorizer = Vectorizer()
    vectorizer.build_index(sample_chunks)
    
    # 创建RAG系统
    rag_system = RAGSystem(vectorizer)
    
    # 测试问答
    questions = [
        "什么是人工智能？",
        "机器学习和人工智能有什么关系？",
        "如何实现机器学习？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        response = rag_system.answer_question(question)
        print(f"答案: {response['answer']}")
        print(f"来源数量: {len(response['sources'])}")
    
    # 获取系统统计
    stats = rag_system.get_system_stats()
    print(f"\n系统统计: {stats}")
