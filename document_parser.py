"""
企业知识库 - 文档解析器
支持PDF、Word、Excel等格式的文档解析
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import paddleocr
from PIL import Image
import numpy as np
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    """文档解析器主类"""
    
    def __init__(self, ocr_lang: str = 'ch'):
        """
        初始化文档解析器
        
        Args:
            ocr_lang: OCR语言，默认中文
        """
        self.ocr_lang = ocr_lang
        self.ocr = None
        self._init_ocr()
        
    def _init_ocr(self):
        """初始化OCR引擎"""
        try:
            self.ocr = paddleocr.PaddleOCR(
                use_angle_cls=True, 
                lang=self.ocr_lang,
                show_log=False
            )
            logger.info("OCR引擎初始化成功")
        except Exception as e:
            logger.warning(f"OCR引擎初始化失败: {e}")
            self.ocr = None
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        解析文档主入口
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            解析结果字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return self.parse_pdf(file_path)
            elif file_ext in ['.doc', '.docx']:
                return self.parse_word(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                return self.parse_excel(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
        except Exception as e:
            logger.error(f"解析文档失败: {e}")
            raise
    
    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        解析PDF文档
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            PDF解析结果
        """
        logger.info(f"开始解析PDF: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            pages = []
            
            for page_num in range(len(doc)):
                logger.info(f"解析第 {page_num + 1} 页")
                page = doc[page_num]
                
                # 提取文本
                text = page.get_text()
                
                # 提取表格
                tables = self.extract_tables_from_page(page)
                
                # 提取图片
                images = self.extract_images_from_page(page, page_num)
                
                # 提取页面信息
                page_info = {
                    'page_num': page_num + 1,
                    'text': text.strip(),
                    'tables': tables,
                    'images': images,
                    'width': page.rect.width,
                    'height': page.rect.height
                }
                
                pages.append(page_info)
            
            doc.close()
            
            result = {
                'file_path': str(pdf_path),
                'file_type': 'pdf',
                'total_pages': len(pages),
                'pages': pages,
                'parse_time': None,  # 可以添加时间戳
                'status': 'success'
            }
            
            logger.info(f"PDF解析完成，共 {len(pages)} 页")
            return result
            
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            raise
    
    def extract_tables_from_page(self, page) -> List[Dict]:
        """
        从页面提取表格
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            表格列表
        """
        tables = []
        
        try:
            # 使用PyMuPDF的表格检测功能
            tab = page.find_tables()
            
            for table in tab:
                table_data = []
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        row_data.append(cell.text.strip())
                    table_data.append(row_data)
                
                if table_data:
                    tables.append({
                        'data': table_data,
                        'bbox': table.bbox,
                        'rows': len(table_data),
                        'cols': len(table_data[0]) if table_data else 0
                    })
            
            # 如果没有检测到表格，尝试OCR识别
            if not tables and self.ocr:
                tables.extend(self._ocr_tables_from_page(page))
                
        except Exception as e:
            logger.warning(f"表格提取失败: {e}")
        
        return tables
    
    def _ocr_tables_from_page(self, page) -> List[Dict]:
        """
        使用OCR识别页面中的表格
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            OCR识别的表格列表
        """
        tables = []
        
        try:
            # 将页面转换为图像
            pix = page.get_pixmap()
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # 转换为RGB格式
            if pix.n == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:  # 灰度
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # 使用OpenCV进行表格检测
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选可能的表格区域
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 面积阈值
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 检查是否为矩形（表格特征）
                    if self._is_table_region(x, y, w, h, gray):
                        # 提取该区域的图像进行OCR
                        roi = img_array[y:y+h, x:x+w]
                        table_text = self._ocr_image_text(roi)
                        
                        if table_text:
                            tables.append({
                                'data': self._parse_table_text(table_text),
                                'bbox': (x, y, w, h),
                                'method': 'ocr'
                            })
            
        except Exception as e:
            logger.warning(f"OCR表格识别失败: {e}")
        
        return tables
    
    def _is_table_region(self, x: int, y: int, w: int, h: int, gray_img) -> bool:
        """
        判断区域是否为表格
        
        Args:
            x, y, w, h: 区域坐标和尺寸
            gray_img: 灰度图像
            
        Returns:
            是否为表格区域
        """
        try:
            roi = gray_img[y:y+h, x:x+w]
            
            # 计算水平线和垂直线
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//10, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//10))
            
            horizontal_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vertical_kernel)
            
            # 计算线条密度
            h_density = np.sum(horizontal_lines > 0) / (w * h)
            v_density = np.sum(vertical_lines > 0) / (w * h)
            
            # 表格应该有较多的水平线和垂直线
            return h_density > 0.01 and v_density > 0.01
            
        except Exception:
            return False
    
    def _ocr_image_text(self, image: np.ndarray) -> str:
        """
        OCR识别图像中的文本
        
        Args:
            image: 图像数组
            
        Returns:
            识别的文本
        """
        if self.ocr is None:
            return ""
        
        try:
            result = self.ocr.ocr(image, cls=True)
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line[1][1] > 0.5]
                return " ".join(texts)
        except Exception as e:
            logger.warning(f"OCR识别失败: {e}")
        
        return ""
    
    def _parse_table_text(self, text: str) -> List[List[str]]:
        """
        解析表格文本为二维数组
        
        Args:
            text: OCR识别的文本
            
        Returns:
            表格数据
        """
        # 简单的表格解析，按行分割
        lines = text.split('\n')
        table_data = []
        
        for line in lines:
            if line.strip():
                # 按空格或制表符分割列
                columns = [col.strip() for col in line.split() if col.strip()]
                if columns:
                    table_data.append(columns)
        
        return table_data
    
    def extract_images_from_page(self, page, page_num: int) -> List[Dict]:
        """
        从页面提取图片
        
        Args:
            page: PyMuPDF页面对象
            page_num: 页面编号
            
        Returns:
            图片信息列表
        """
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # 生成图片文件名
                    img_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    img_path = f"temp_images/{img_filename}"
                    
                    # 确保目录存在
                    os.makedirs("temp_images", exist_ok=True)
                    
                    # 保存图片
                    with open(img_path, "wb") as image_file:
                        image_file.write(image_bytes)
                    
                    images.append({
                        'filename': img_filename,
                        'path': img_path,
                        'size': len(image_bytes),
                        'format': image_ext,
                        'bbox': img[1:5] if len(img) > 4 else None
                    })
                    
                except Exception as e:
                    logger.warning(f"提取图片失败: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"页面图片提取失败: {e}")
        
        return images
    
    def parse_word(self, doc_path: Path) -> Dict[str, Any]:
        """
        解析Word文档
        
        Args:
            doc_path: Word文档路径
            
        Returns:
            Word解析结果
        """
        # TODO: 实现Word文档解析
        logger.info(f"Word文档解析功能待实现: {doc_path}")
        return {
            'file_path': str(doc_path),
            'file_type': 'word',
            'status': 'not_implemented'
        }
    
    def parse_excel(self, excel_path: Path) -> Dict[str, Any]:
        """
        解析Excel文档
        
        Args:
            excel_path: Excel文档路径
            
        Returns:
            Excel解析结果
        """
        # TODO: 实现Excel文档解析
        logger.info(f"Excel文档解析功能待实现: {excel_path}")
        return {
            'file_path': str(excel_path),
            'file_type': 'excel',
            'status': 'not_implemented'
        }
    
    def save_parse_result(self, result: Dict[str, Any], output_path: str):
        """
        保存解析结果到JSON文件
        
        Args:
            result: 解析结果
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"解析结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存解析结果失败: {e}")
            raise

# 使用示例
if __name__ == "__main__":
    # 创建解析器
    parser = DocumentParser()
    
    # 解析PDF文档
    try:
        result = parser.parse_document("example.pdf")
        
        # 保存结果
        parser.save_parse_result(result, "parse_result.json")
        
        print(f"解析完成，共 {result['total_pages']} 页")
        print(f"发现 {sum(len(page['tables']) for page in result['pages'])} 个表格")
        print(f"发现 {sum(len(page['images']) for page in result['pages'])} 张图片")
        
    except Exception as e:
        print(f"解析失败: {e}")
