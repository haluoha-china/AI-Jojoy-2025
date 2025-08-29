"""
企业知识库 - 文本分块器
实现智能的文本分割，支持多种分割策略
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SplitStrategy(Enum):
    """分割策略枚举"""
    SENTENCE = "sentence"      # 按句子分割
    PARAGRAPH = "paragraph"    # 按段落分割
    PAGE = "page"             # 按页面分割
    FIXED_SIZE = "fixed_size"  # 按固定大小分割
    SEMANTIC = "semantic"      # 按语义分割

@dataclass
class TextChunk:
    """文本块数据结构"""
    content: str               # 文本内容
    chunk_id: str             # 块ID
    page_num: Optional[int]   # 页码
    chunk_type: str           # 块类型
    metadata: Dict[str, Any]  # 元数据
    start_pos: Optional[int]  # 在原文中的起始位置
    end_pos: Optional[int]    # 在原文中的结束位置

class TextSplitter:
    """文本分块器主类"""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 strategy: SplitStrategy = SplitStrategy.SENTENCE):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 重叠大小（字符数）
            strategy: 分割策略
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # 中文句子分割正则表达式
        self.chinese_sentence_pattern = r'[。！？；\n\r]+'
        
        # 段落分割正则表达式
        self.paragraph_pattern = r'\n\s*\n+'
        
        # 标题识别正则表达式
        self.title_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节条]',  # 第X章、第X节等
            r'^[一二三四五六七八九十\d]+[、．.]',     # 1、2、等
            r'^[A-Z][A-Z\s]*$',                    # 全大写字母
            r'^\d+\.',                             # 1. 2. 等
        ]
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        分割文本
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            文本块列表
        """
        if not text or not text.strip():
            return []
        
        if metadata is None:
            metadata = {}
        
        if self.strategy == SplitStrategy.SENTENCE:
            return self._split_by_sentences(text, metadata)
        elif self.strategy == SplitStrategy.PARAGRAPH:
            return self._split_by_paragraphs(text, metadata)
        elif self.strategy == SplitStrategy.PAGE:
            return self._split_by_pages(text, metadata)
        elif self.strategy == SplitStrategy.FIXED_SIZE:
            return self._split_by_fixed_size(text, metadata)
        elif self.strategy == SplitStrategy.SEMANTIC:
            return self._split_by_semantic(text, metadata)
        else:
            raise ValueError(f"不支持的分割策略: {self.strategy}")
    
    def _split_by_sentences(self, text: str, metadata: Dict) -> List[TextChunk]:
        """按句子分割文本"""
        chunks = []
        
        # 使用正则表达式分割句子
        sentences = re.split(self.chinese_sentence_pattern, text)
        
        current_chunk = ""
        chunk_start = 0
        chunk_id = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查是否需要开始新的块
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # 保存当前块
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    page_num=metadata.get('page_num'),
                    chunk_type='sentence',
                    metadata=metadata.copy(),
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # 开始新块，包含重叠部分
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + sentence
                chunk_start = chunk_start + len(current_chunk) - len(overlap_text)
                chunk_id += 1
            else:
                current_chunk += sentence + "。"
        
        # 添加最后一个块
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                page_num=metadata.get('page_num'),
                chunk_type='sentence',
                metadata=metadata.copy(),
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str, metadata: Dict) -> List[TextChunk]:
        """按段落分割文本"""
        chunks = []
        
        # 分割段落
        paragraphs = re.split(self.paragraph_pattern, text)
        
        current_chunk = ""
        chunk_start = 0
        chunk_id = 1
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 检查是否需要开始新的块
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # 保存当前块
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    page_num=metadata.get('page_num'),
                    chunk_type='paragraph',
                    metadata=metadata.copy(),
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # 开始新块
                current_chunk = paragraph
                chunk_start = chunk_start + len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # 添加最后一个块
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                page_num=metadata.get('page_num'),
                chunk_type='paragraph',
                metadata=metadata.copy(),
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_pages(self, text: str, metadata: Dict) -> List[TextChunk]:
        """按页面分割文本（假设文本已经按页面组织）"""
        # 这里假设输入的text是单个页面的内容
        chunk = TextChunk(
            content=text.strip(),
            chunk_id=f"page_{metadata.get('page_num', 1)}",
            page_num=metadata.get('page_num'),
            chunk_type='page',
            metadata=metadata.copy(),
            start_pos=0,
            end_pos=len(text)
        )
        return [chunk]
    
    def _split_by_fixed_size(self, text: str, metadata: Dict) -> List[TextChunk]:
        """按固定大小分割文本"""
        chunks = []
        chunk_id = 1
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_content = text[i:i + self.chunk_size]
            
            if chunk_content.strip():
                chunk = TextChunk(
                    content=chunk_content.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    page_num=metadata.get('page_num'),
                    chunk_type='fixed_size',
                    metadata=metadata.copy(),
                    start_pos=i,
                    end_pos=min(i + self.chunk_size, len(text))
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _split_by_semantic(self, text: str, metadata: Dict) -> List[TextChunk]:
        """按语义分割文本（基于标题和结构）"""
        chunks = []
        
        # 按行分割
        lines = text.split('\n')
        
        current_chunk = ""
        chunk_start = 0
        chunk_id = 1
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 检查是否为标题
            is_title = self._is_title(line)
            
            # 如果遇到标题且当前块不为空，保存当前块
            if is_title and current_chunk and len(current_chunk) > self.chunk_size // 2:
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    page_num=metadata.get('page_num'),
                    chunk_type='semantic',
                    metadata=metadata.copy(),
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                current_chunk = line
                chunk_start = chunk_start + len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += "\n" + line if current_chunk else line
                
                # 如果当前块过大，强制分割
                if len(current_chunk) > self.chunk_size:
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=f"chunk_{chunk_id}",
                        page_num=metadata.get('page_num'),
                        chunk_type='semantic',
                        metadata=metadata.copy(),
                        start_pos=chunk_start,
                        end_pos=chunk_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    
                    current_chunk = ""
                    chunk_start = chunk_start + len(current_chunk)
                    chunk_id += 1
        
        # 添加最后一个块
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                page_num=metadata.get('page_num'),
                chunk_type='semantic',
                metadata=metadata.copy(),
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _is_title(self, line: str) -> bool:
        """判断是否为标题行"""
        line = line.strip()
        
        # 检查各种标题模式
        for pattern in self.title_patterns:
            if re.match(pattern, line):
                return True
        
        # 检查长度和格式特征
        if len(line) < 50 and (line.isupper() or line.endswith('：') or line.endswith(':')):
            return True
        
        return False
    
    def split_document_pages(self, pages: List[Dict]) -> List[TextChunk]:
        """
        分割文档页面列表
        
        Args:
            pages: 页面列表，每个页面包含text等字段
            
        Returns:
            文本块列表
        """
        all_chunks = []
        
        for page in pages:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # 为每个页面创建元数据
            page_metadata = {
                'page_num': page.get('page_num'),
                'file_path': page.get('file_path', ''),
                'tables_count': len(page.get('tables', [])),
                'images_count': len(page.get('images', [])),
                'page_width': page.get('width'),
                'page_height': page.get('height')
            }
            
            # 分割当前页面
            page_chunks = self.split_text(page_text, page_metadata)
            all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def optimize_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        优化文本块
        
        Args:
            chunks: 原始文本块列表
            
        Returns:
            优化后的文本块列表
        """
        optimized_chunks = []
        
        for chunk in chunks:
            # 清理内容
            cleaned_content = self._clean_chunk_content(chunk.content)
            
            # 如果清理后内容太短，跳过
            if len(cleaned_content) < 50:
                continue
            
            # 创建优化后的块
            optimized_chunk = TextChunk(
                content=cleaned_content,
                chunk_id=chunk.chunk_id,
                page_num=chunk.page_num,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata.copy(),
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos
            )
            
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """清理文本块内容"""
        if not content:
            return ""
        
        # 移除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        
        # 移除行首行尾空白
        content = content.strip()
        
        # 移除特殊字符（可选）
        # content = re.sub(r'[^\w\s\u4e00-\u9fff。！？；，：""''（）【】]', '', content)
        
        return content
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        获取文本块统计信息
        
        Args:
            chunks: 文本块列表
            
        Returns:
            统计信息字典
        """
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_chars = sum(len(chunk.content) for chunk in chunks)
        
        # 按类型统计
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        # 按页面统计
        page_counts = {}
        for chunk in chunks:
            page_num = chunk.page_num
            if page_num:
                page_counts[page_num] = page_counts.get(page_num, 0) + 1
        
        # 计算平均块大小
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'chunk_types': type_counts,
            'pages_coverage': len(page_counts),
            'page_distribution': page_counts
        }

# 使用示例
if __name__ == "__main__":
    # 创建分块器
    splitter = TextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        strategy=SplitStrategy.SENTENCE
    )
    
    # 示例文本
    sample_text = """
    第一章 企业概述
    
    我们公司成立于2010年，是一家专注于人工智能技术研发的高新技术企业。
    公司总部位于北京，在上海、深圳等地设有分支机构。
    
    第二章 技术优势
    
    公司在自然语言处理、计算机视觉、机器学习等领域拥有深厚的技术积累。
    我们拥有多项自主知识产权，并与多家知名高校建立了合作关系。
    
    第三章 发展前景
    
    随着人工智能技术的快速发展，公司面临着巨大的发展机遇。
    我们将继续加大研发投入，推动技术创新，为客户提供更好的产品和服务。
    """
    
    # 分割文本
    chunks = splitter.split_text(sample_text, {'source': 'sample'})
    
    print(f"分割完成，共生成 {len(chunks)} 个文本块：")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1} (ID: {chunk.chunk_id}):")
        print(f"内容: {chunk.content[:100]}...")
        print(f"类型: {chunk.chunk_type}")
        print(f"长度: {len(chunk.content)} 字符")
    
    # 获取统计信息
    stats = splitter.get_chunk_statistics(chunks)
    print(f"\n统计信息: {stats}")
