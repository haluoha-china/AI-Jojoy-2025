"""
企业知识库 - 主Flask应用
提供Web界面和API接口
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback

# 导入自定义模块
from document_parser import DocumentParser
from text_splitter import TextSplitter, SplitStrategy
from vectorizer import Vectorizer
from rag_system import RAGSystem
from glossary_service import GlossaryService
from qwen_agent_integration_enhanced import QwenAgentIntegrationEnhanced

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB文件大小限制

# 全局变量
document_parser = None
text_splitter = None
vectorizer = None
rag_system = None
glossary_service = None
qwen_agent_integration = None

# 配置
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp_images'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
INDEX_FOLDER = 'indexes'

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_system():
    """初始化系统组件"""
    global document_parser, text_splitter, vectorizer, rag_system, glossary_service
    
    try:
        # 初始化文档解析器
        document_parser = DocumentParser()
        logger.info("文档解析器初始化完成")
        
        # 初始化文本分块器
        text_splitter = TextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            strategy=SplitStrategy.SENTENCE
        )
        logger.info("文本分块器初始化完成")
        
        # 初始化向量化器
        vectorizer = Vectorizer()
        logger.info("向量化器初始化完成")
        
        # 初始化术语对照服务
        glossary_service = GlossaryService()
        logger.info("术语对照服务初始化完成")
        
        # 初始化千问Agent集成系统
        qwen_agent_integration = QwenAgentIntegrationEnhanced()
        logger.info("千问Agent集成系统初始化完成")

        # 尝试加载已有索引
        index_path = os.path.join(INDEX_FOLDER, 'latest')
        if os.path.exists(index_path + '.faiss'):
            try:
                vectorizer.load_index(index_path)
                logger.info("已加载现有索引")
                
                # 初始化RAG系统
                rag_system = RAGSystem(vectorizer)
                logger.info("RAG系统初始化完成")
                
            except Exception as e:
                logger.warning(f"加载现有索引失败: {e}")
                logger.info("将创建新的索引")
        
        logger.info("系统初始化完成")
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    """上传文档"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        file.save(filepath)
        logger.info(f"文件上传成功: {filepath}")
        
        # 解析文档
        if not document_parser:
            return jsonify({'error': '系统未初始化'}), 500
        
        parse_result = document_parser.parse_document(filepath)
        
        # 文本分块
        if not text_splitter:
            return jsonify({'error': '文本分块器未初始化'}), 500
        
        chunks = text_splitter.split_document_pages(parse_result['pages'])
        
        # 优化文本块
        optimized_chunks = text_splitter.optimize_chunks(chunks)
        
        # 构建向量索引
        if not vectorizer:
            return jsonify({'error': '向量化器未初始化'}), 500
        
        vectorizer.build_index(optimized_chunks)
        
        # 初始化RAG系统
        global rag_system
        rag_system = RAGSystem(vectorizer)
        
        # 保存索引
        index_path = os.path.join(INDEX_FOLDER, 'latest')
        vectorizer.save_index(index_path)
        
        # 获取统计信息
        chunk_stats = text_splitter.get_chunk_statistics(optimized_chunks)
        index_info = vectorizer.get_index_info()
        
        return jsonify({
            'message': '文档上传和解析成功',
            'filename': filename,
            'file_path': filepath,
            'parse_result': {
                'total_pages': parse_result['total_pages'],
                'tables_count': sum(len(page.get('tables', [])) for page in parse_result['pages']),
                'images_count': sum(len(page.get('images', [])) for page in parse_result['pages'])
            },
            'chunks_info': chunk_stats,
            'index_info': index_info
        })
        
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'文档上传失败: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """提问"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        # 获取可选参数
        top_k = data.get('top_k', 5)
        score_threshold = data.get('score_threshold', 0.3)
        include_context = data.get('include_context', True)
        model = data.get('model', 'qwen')  # qwen 或 local
        
        if not rag_system:
            return jsonify({'error': '请先上传文档'}), 400
        
        # 术语规范化
        normalized_question = question
        glossary_hits = {}
        if glossary_service:
            normalized_question, glossary_hits = glossary_service.normalize_question(question)

        # 获取答案（使用规范化后的问题）
        response = rag_system.answer_question(
            normalized_question, 
            top_k=top_k, 
            score_threshold=score_threshold,
            include_context=include_context
        )
        
        # 附加术语命中与模型选择信息
        response['normalized_question'] = normalized_question
        response['glossary_hits'] = glossary_hits
        response['model'] = model

        return jsonify(response)
        
    except Exception as e:
        logger.error(f"问题处理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'问题处理失败: {str(e)}'}), 500

@app.route('/batch_ask', methods=['POST'])
def batch_ask():
    """批量提问"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
        
        questions = data.get('questions', [])
        if not questions or not isinstance(questions, list):
            return jsonify({'error': '问题列表不能为空'}), 400
        
        # 获取可选参数
        top_k = data.get('top_k', 5)
        score_threshold = data.get('score_threshold', 0.3)
        
        if not rag_system:
            return jsonify({'error': '请先上传文档'}), 400
        
        # 批量回答问题
        responses = rag_system.batch_answer_questions(
            questions, 
            top_k=top_k, 
            score_threshold=score_threshold
        )
        
        return jsonify({
            'questions': questions,
            'responses': responses,
            'total_questions': len(questions),
            'processed_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"批量问题处理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'批量问题处理失败: {str(e)}'}), 500

@app.route('/system_info', methods=['GET'])
def get_system_info():
    """获取系统信息"""
    try:
        info = {
            'system_status': 'running',
            'initialization_time': None,
            'components': {}
        }
        
        if document_parser:
            info['components']['document_parser'] = 'initialized'
        
        if text_splitter:
            info['components']['text_splitter'] = 'initialized'
        
        if vectorizer:
            info['components']['vectorizer'] = 'initialized'
            index_info = vectorizer.get_index_info()
            info['vector_index'] = index_info
        
        if rag_system:
            info['components']['rag_system'] = 'initialized'
            rag_stats = rag_system.get_system_stats()
            info['rag_stats'] = rag_stats
        
        if qwen_agent_integration:
            info['components']['qwen_agent_integration'] = 'initialized'
            info['qwen_agent_status'] = 'ready'
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        return jsonify({'error': f'获取系统信息失败: {str(e)}'}), 500

@app.route('/conversation_history', methods=['GET'])
def get_conversation_history():
    """获取对话历史"""
    try:
        if not rag_system:
            return jsonify({'error': 'RAG系统未初始化'}), 400
        
        limit = request.args.get('limit', type=int)
        history = rag_system.get_conversation_history(limit)
        
        return jsonify({
            'history': history,
            'total_count': len(history)
        })
        
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        return jsonify({'error': f'获取对话历史失败: {str(e)}'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空对话历史"""
    try:
        if not rag_system:
            return jsonify({'error': 'RAG系统未初始化'}), 400
        
        rag_system.clear_history()
        
        return jsonify({'message': '对话历史已清空'})
        
    except Exception as e:
        logger.error(f"清空对话历史失败: {e}")
        return jsonify({'error': f'清空对话历史失败: {str(e)}'}), 500

@app.route('/export_history', methods=['POST'])
def export_history():
    """导出对话历史"""
    try:
        if not rag_system:
            return jsonify({'error': 'RAG系统未初始化'}), 400
        
        data = request.get_json() or {}
        format_type = data.get('format', 'json')
        
        # 生成导出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"conversation_history_{timestamp}.{format_type}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 导出历史
        rag_system.export_conversation_history(filepath, format_type)
        
        return jsonify({
            'message': '对话历史导出成功',
            'filename': filename,
            'filepath': filepath
        })
        
    except Exception as e:
        logger.error(f"导出对话历史失败: {e}")
        return jsonify({'error': f'导出对话历史失败: {str(e)}'}), 500

@app.route('/chunks', methods=['GET'])
def get_chunks():
    """获取文本块信息"""
    try:
        if not vectorizer:
            return jsonify({'error': '向量化器未初始化'}), 400
        
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        chunk_type = request.args.get('chunk_type', '')
        page_num = request.args.get('page_num', type=int)
        
        chunks = vectorizer.chunks
        
        # 过滤
        if chunk_type:
            chunks = [c for c in chunks if c.get('chunk_type') == chunk_type]
        
        if page_num:
            chunks = [c for c in chunks if c.get('page_num') == page_num]
        
        # 分页
        total = len(chunks)
        start = (page - 1) * per_page
        end = start + per_page
        page_chunks = chunks[start:end]
        
        return jsonify({
            'chunks': page_chunks,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        logger.error(f"获取文本块信息失败: {e}")
        return jsonify({'error': f'获取文本块信息失败: {str(e)}'}), 500

@app.route('/search_chunks', methods=['POST'])
def search_chunks():
    """搜索文本块"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': '搜索查询不能为空'}), 400
        
        top_k = data.get('top_k', 10)
        score_threshold = data.get('score_threshold', 0.0)
        
        if not vectorizer:
            return jsonify({'error': '向量化器未初始化'}), 400
        
        # 搜索
        results = vectorizer.search(query, top_k, score_threshold)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_found': len(results)
        })
        
    except Exception as e:
        logger.error(f"搜索文本块失败: {e}")
        return jsonify({'error': f'搜索文本块失败: {str(e)}'}), 500

@app.route('/files', methods=['GET'])
def get_uploaded_files():
    """获取已上传的文件列表"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                files.append({
                    'filename': filename,
                    'size': stat.st_size,
                    'upload_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': filename.rsplit('.', 1)[1].lower() if '.' in filename else 'unknown'
                })
        
        # 按上传时间排序
        files.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return jsonify({
            'files': files,
            'total_count': len(files)
        })
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        return jsonify({'error': f'获取文件列表失败: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下载文件"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"文件下载失败: {e}")
        return jsonify({'error': f'文件下载失败: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'document_parser': document_parser is not None,
            'text_splitter': text_splitter is not None,
            'vectorizer': vectorizer is not None,
            'rag_system': rag_system is not None,
            'qwen_agent_integration': qwen_agent_integration is not None
        }
    })

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({'error': '接口不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"内部服务器错误: {error}")
    return jsonify({'error': '内部服务器错误'}), 500

@app.errorhandler(413)
def too_large(error):
    """文件过大错误处理"""
    return jsonify({'error': '文件过大，请上传小于100MB的文件'}), 413

# ==================== 千问Agent Function Call 接口 ====================
# 按照千问Agent真实工作方式设计：先术语翻译，再RAG检索，最后生成答案

@app.route('/qwen/term_lookup', methods=['POST'])
def qwen_term_lookup():
    """千问Agent Function Call: 术语翻译"""
    try:
        if not qwen_agent_integration:
            return jsonify({'error': '千问Agent集成系统未初始化'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
        
        term = data.get('term', '').strip()
        if not term:
            return jsonify({'error': '术语不能为空'}), 400
        
        # 调用术语翻译Function Call
        result = qwen_agent_integration.fc_normalize(term)
        return jsonify({
            'term': term,
            'result': result['normalized_question'],
            'glossary_hits': result['glossary_hits'],
            'has_changes': result['has_changes'],
            'timestamp': result['timestamp'],
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"术语翻译Function Call失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'术语翻译失败: {str(e)}'}), 500

@app.route('/qwen/rag_search', methods=['POST'])
def qwen_rag_search():
    """千问Agent Function Call: 知识库检索"""
    try:
        if not qwen_agent_integration:
            return jsonify({'error': '千问Agent集成系统未初始化'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': '查询不能为空'}), 400
        
        top_k = data.get('top_k', 5)
        
        # 调用知识库检索Function Call
        result = qwen_agent_integration.fc_retrieve(query, top_k=top_k)
        return jsonify({
            'query': query,
            'result': result['search_results'],
            'total_results': result['total_results'],
            'top_k': result['top_k'],
            'timestamp': result['timestamp'],
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"知识库检索Function Call失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'知识库检索失败: {str(e)}'}), 500



@app.route('/qwen/fc_get_status', methods=['GET'])
def qwen_fc_get_status():
    """千问Agent Function Call: 获取系统状态"""
    try:
        if not qwen_agent_integration:
            return jsonify({'error': '千问Agent集成系统未初始化'}), 400
        
        # 获取系统状态
        status = {
            'system_status': 'running',
            'qwen_agent_integration': 'ready',
            'glossary_service': 'ready' if glossary_service else 'not_initialized',
            'knowledge_base': 'ready' if qwen_agent_integration.kb else 'not_initialized',
            'timestamp': datetime.now().isoformat(),
            'function_calls': {
                'term_lookup': '/qwen/term_lookup',
                'rag_search': '/qwen/rag_search', 
                'fc_get_status': '/qwen/fc_get_status'
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取系统状态Function Call失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'获取系统状态失败: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        # 初始化系统
        initialize_system()
        
        # 启动Flask应用
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        logger.error(traceback.format_exc())
