"""
术语对照服务
从企业黑话表（JSON或Excel）加载缩略语与标准术语映射，支持问题规范化。
"""

import os
import re
import json
import logging
from typing import Dict, Tuple, List, Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlossaryService:
    """加载并提供术语规范化能力。"""

    def __init__(self, json_path: str = "enterprise_terminology_complete.json", 
                 excel_path: str = "公司常用缩略语20250401.xlsx"):
        self.json_path = json_path
        self.excel_path = excel_path
        self.term_to_standard: Dict[str, str] = {}
        self._load_glossary()

    def _load_glossary(self) -> None:
        """优先从JSON加载术语映射，如果没有则从Excel加载。
        
        兼容常见列名：
        - 缩略语/简称/术语 -> term
        - 全称/标准术语/说明 -> standard
        """
        # 优先尝试JSON文件
        if os.path.exists(self.json_path):
            try:
                self._load_from_json()
                return
            except Exception as e:
                logger.warning(f"JSON文件加载失败: {e}，尝试Excel文件")
        
        # 回退到Excel文件
        if os.path.exists(self.excel_path):
            try:
                self._load_from_excel()
                return
            except Exception as e:
                logger.warning(f"Excel文件加载失败: {e}")
        
        # 如果都失败了
        logger.warning(f"未找到可用的术语表文件，术语规范化将被跳过")
        logger.info(f"尝试的文件: {self.json_path}, {self.excel_path}")

    def _load_from_json(self) -> None:
        """从JSON文件加载术语映射"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理不同的JSON结构
            if isinstance(data, list):
                # 如果是列表格式
                mapping = {}
                for item in data:
                    if isinstance(item, dict):
                        # 尝试不同的键名
                        term = item.get('term') or item.get('abbr') or item.get('short') or item.get('key')
                        standard = item.get('standard') or item.get('full') or item.get('explanation') or item.get('value')
                        
                        if term and standard:
                            mapping[str(term).strip()] = str(standard).strip()
                
                self.term_to_standard = mapping
                
            elif isinstance(data, dict):
                # 如果是字典格式
                mapping = {}
                for term, standard in data.items():
                    if isinstance(standard, str):
                        mapping[str(term).strip()] = standard.strip()
                    elif isinstance(standard, dict):
                        # 如果值是字典，尝试提取标准术语
                        std_value = standard.get('standard') or standard.get('full') or standard.get('explanation')
                        if std_value:
                            mapping[str(term).strip()] = str(std_value).strip()
                
                self.term_to_standard = mapping
            
            # 同时加入大写/小写映射
            final_mapping = {}
            for k, v in self.term_to_standard.items():
                final_mapping[k] = v
                final_mapping[k.upper()] = v
                final_mapping[k.lower()] = v
            
            self.term_to_standard = final_mapping
            
            logger.info(f"✅ 从JSON文件加载术语表成功，共 {len(self.term_to_standard)} 条映射")
            
        except Exception as e:
            logger.error(f"JSON文件加载失败: {e}")
            raise

    def _load_from_excel(self) -> None:
        """从Excel文件加载术语映射"""
        try:
            df = pd.read_excel(self.excel_path)
            columns = {c.strip().lower(): c for c in df.columns if isinstance(c, str)}

            # 猜测列名
            term_col = None
            for k, v in columns.items():
                if any(x in k for x in ["缩略", "简称", "术语", "term", "abbr", "关键词"]):
                    term_col = v
                    break
            if term_col is None:
                term_col = df.columns[0]

            std_col = None
            for k, v in columns.items():
                if any(x in k for x in ["全称", "标准", "说明", "释义", "含义", "standard", "expand"]):
                    std_col = v
                    break
            if std_col is None and len(df.columns) > 1:
                std_col = df.columns[1]
            elif std_col is None:
                std_col = term_col

            mapping = {}
            for _, row in df.iterrows():
                term = str(row.get(term_col, "")).strip()
                standard = str(row.get(std_col, "")).strip()
                if not term:
                    continue
                if not standard:
                    standard = term
                mapping[term] = standard

            # 同时加入大写/小写映射
            self.term_to_standard = {}
            for k, v in mapping.items():
                self.term_to_standard[k] = v
                self.term_to_standard[k.upper()] = v
                self.term_to_standard[k.lower()] = v

            logger.info(f"✅ 从Excel文件加载术语表成功，共 {len(self.term_to_standard)} 条映射")
            
        except Exception as e:
            logger.error(f"Excel文件加载失败: {e}")
            raise

    def normalize_question(self, question: str) -> Tuple[str, Dict[str, str]]:
        """将问题中的缩略语替换为标准术语。

        返回 (规范化后的问题, 命中的术语映射)
        """
        if not question or not self.term_to_standard:
            return question, {}

        hits: Dict[str, str] = {}
        normalized = question

        # 为避免误替换，按词边界替换，优先匹配长词
        terms_sorted: List[str] = sorted(self.term_to_standard.keys(), key=len, reverse=True)
        for term in terms_sorted:
            standard = self.term_to_standard[term]
            # 使用大小写不敏感匹配，词边界或非中文/字母数字边界
            pattern = r"(?i)(?<![\w\-\uffff])" + re.escape(term) + r"(?![\w\-\uffff])"
            if re.search(pattern, normalized):
                normalized = re.sub(pattern, standard, normalized)
                hits[term] = standard

        return normalized, hits

    def get_glossary_info(self) -> Dict[str, Any]:
        """获取术语表信息"""
        return {
            'source_file': self.json_path if os.path.exists(self.json_path) else self.excel_path,
            'total_terms': len(self.term_to_standard),
            'sample_terms': list(self.term_to_standard.keys())[:10] if self.term_to_standard else []
        }


