#!/usr/bin/env python3
"""
千问Agent Function Schema定义
- 为企业知识库系统定义标准的Function Call接口
- 支持术语规范化、知识库检索、端到端处理等功能
- 符合千问Agent的标准Function Call格式
"""

from typing import Dict, Any, List

# ========== Function Schema 定义 ==========

# 1. 术语规范化函数
NORMALIZE_QUESTION_SCHEMA = {
    "name": "normalize_question",
    "description": "将用户问题中的企业术语转换为标准术语，提高知识库检索准确性",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用户输入的原始问题，可能包含企业黑话或非标准术语"
            }
        },
        "required": ["question"]
    }
}

# 2. 知识库检索函数
SEARCH_KNOWLEDGE_SCHEMA = {
    "name": "search_knowledge",
    "description": "在1914个企业文档块中检索相关内容，返回最相关的文档信息",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询词，可以是规范化后的术语或用户原始问题"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认5个，最大10个",
                "default": 5,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["query"]
    }
}

# 3. 端到端处理函数
PROCESS_QUESTION_SCHEMA = {
    "name": "process_question",
    "description": "完整的问答流程：术语规范化 → 知识库检索 → 结果汇总",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用户输入的原始问题"
            },
            "top_k": {
                "type": "integer",
                "description": "检索结果数量，默认5个",
                "default": 5,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["question"]
    }
}

# 4. 初始响应函数
INITIAL_RESPONSE_SCHEMA = {
    "name": "initial_response",
    "description": "为用户提供问题分析和模型选择建议，不直接生成答案",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用户输入的原始问题"
            }
        },
        "required": ["question"]
    }
}

# ========== 函数映射表 ==========

FUNCTION_SCHEMAS = {
    "normalize_question": NORMALIZE_QUESTION_SCHEMA,
    "search_knowledge": SEARCH_KNOWLEDGE_SCHEMA,
    "process_question": PROCESS_QUESTION_SCHEMA,
    "initial_response": INITIAL_RESPONSE_SCHEMA
}

# ========== 函数调用映射 ==========

FUNCTION_MAPPING = {
    "normalize_question": "fc_normalize",
    "search_knowledge": "fc_retrieve", 
    "process_question": "fc_process",
    "initial_response": "initial_response"
}

# ========== 使用说明 ==========

def get_function_schemas() -> List[Dict[str, Any]]:
    """获取所有Function Schema列表，供千问Agent使用"""
    return list(FUNCTION_SCHEMAS.values())

def get_function_schema(name: str) -> Dict[str, Any]:
    """获取指定名称的Function Schema"""
    return FUNCTION_SCHEMAS.get(name, {})

def get_function_mapping() -> Dict[str, str]:
    """获取函数名称到实际方法的映射"""
    return FUNCTION_MAPPING.copy()

# ========== 示例用法 ==========

if __name__ == "__main__":
    print("🧪 千问Agent Function Schema 测试\n")
    
    print("1) 所有可用的Function Schema:")
    schemas = get_function_schemas()
    for i, schema in enumerate(schemas, 1):
        print(f"   {i}. {schema['name']}: {schema['description']}")
    
    print("\n2) 函数映射关系:")
    mapping = get_function_mapping()
    for func_name, method_name in mapping.items():
        print(f"   {func_name} -> {method_name}")
    
    print("\n3) 术语规范化Schema示例:")
    norm_schema = get_function_schema("normalize_question")
    print(f"   参数: {norm_schema['parameters']}")
    
    print("\n✅ Function Schema定义完成！")
