from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from bert_classification import  BERTClassifier
import json
from collections import deque, Counter
import os
import re
from nltk.corpus import stopwords
import nltk
from reddit_emotion_analyzer import RedditEmotionAnalyzer
import threading
import queue
import time
from arima_emotion_predictor import SimpleARIMAPredictor
from reddit_emotion_prediction import EmotionPredictor

app = Flask(__name__)

# 全局变量用于存储分析任务
analysis_tasks = {}
task_queue = queue.Queue()

# 全局变量用于存储模型实例（单例模式）
_emotion_classifier = None
_emotion_analyzer = None
_emotion_predictor = None

def get_emotion_classifier():
    """获取情感分类器实例（单例模式）"""
    global _emotion_classifier
    if _emotion_classifier is None:
        model_name = 'google-bert/bert-base-uncased'
        _emotion_classifier = BERTClassifier(num_labels=28, model_name=model_name)
        classifier_path = "model/bert_emotion_classifier_bs_16_lr_2e-05_use_grouped_emotions_28.pt"
        if os.path.exists(classifier_path):
            _emotion_classifier.load_model(classifier_path)
            print(f"情感分类器已加载: {classifier_path}")
        else:
            print(f"警告: 情感分类器模型文件不存在: {classifier_path}")
    return _emotion_classifier

def get_emotion_analyzer():
    """获取情感分析器实例（单例模式）"""
    global _emotion_analyzer
    if _emotion_analyzer is None:
        classifier = get_emotion_classifier()
        _emotion_analyzer = RedditEmotionAnalyzer.get_analyzer(classifier=classifier)
    return _emotion_analyzer

# 下载必要的NLTK数据
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 存储对话历史
conversation_history = deque(maxlen=10)  # 最多保存10条历史记录
emotion_history = deque(maxlen=10)  # 存储情绪历史

# 情绪标签映射（示例，需要根据实际训练数据调整）
emotion_labels_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
# 调整映射以匹配新的emotion_labels_7顺序
# 原映射: [2,2,0,0,2,2,5,5,2,4,0,3,4,2,1,2,4,2,2,1,2,2,5,2,4,4,5,6]
# 新顺序: ["joy", "neutral", "anger", "sadness", "fear", "disgust", "surprise"]
# 转换: anger(0→2), fear(1→4), joy(2→0), disgust(3→5), sadness(4→3), surprise(5→6), neutral(6→1)
MAPPING_7 = [0,0,2,2,0,0,6,6,0,3,2,5,3,0,4,0,3,0,0,4,0,0,6,0,3,3,6,1]
# 修改为与前端一致的情感标签顺序
emotion_labels_7 = [
    "joy", "neutral", "anger", "sadness", "fear", "disgust", "surprise"
]

def reset_history():
    """重置历史记录"""
    global conversation_history, emotion_history
    conversation_history.clear()
    emotion_history.clear()
    print("历史记录已重置")

def load_subreddit_data():
    """加载subreddit数据"""
    try:
        with open('train_by_subreddit_deduplicate.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载subreddit数据失败: {e}")
        return None

@app.route('/')
def home():
    reset_history()  # 每次访问主页时重置历史记录
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print("收到分析请求")  # 调试信息
    text = request.json['text']
    print(f"输入文本: {text}")  # 调试信息
    
    # 获取情感分析器实例
    analyzer = get_emotion_analyzer()
    
    # 使用分析器的classifier进行预测
    pred_labels, probs = analyzer.classifier.predict(text)
    print(f"预测概率: {probs}")  # 调试信息
    
    # 获取top5情绪
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_emotions = [(emotion_labels_28[i], float(probs[i])) for i in top5_indices]
    # 对top5情绪的概率进行归一化
    total_prob = sum(prob for _, prob in top5_emotions)
    top5_emotions = [(emotion, prob/total_prob) for emotion, prob in top5_emotions]
    print(f"Top 5情绪: {top5_emotions}")  # 调试信息
    
    # 更新历史记录
    conversation_history.append(text)
    emotion_history.append(probs)
    print(f"当前历史记录数量: {len(emotion_history)}")  # 调试信息
    
    # 计算平均情绪分布
    avg_emotions = calculate_average_emotions()
    print(f"平均情绪: {avg_emotions}")  # 调试信息
    
    response_data = {
        'text': text,
        'top5_emotions': top5_emotions,
        'conversation_history': list(conversation_history),
        'avg_emotions': avg_emotions
    }
    print(f"返回数据: {response_data}")  # 调试信息
    
    return jsonify(response_data)

def calculate_average_emotions():
    if not emotion_history:
        print("没有情绪历史记录")  # 调试信息
        return []
    
    # 将所有情绪概率向量转换为numpy数组
    emotion_arrays = [np.array(emotions) for emotions in emotion_history]
    print(f"情绪数组数量: {len(emotion_arrays)}")  # 调试信息
    
    # 计算平均值
    avg_probs = np.mean(emotion_arrays, axis=0)
    print(f"平均概率: {avg_probs}")  # 调试信息
    
    # 获取top5情绪
    top5_indices = np.argsort(avg_probs)[-5:][::-1]
    avg_emotions = [(emotion_labels_28[i], float(avg_probs[i])) for i in top5_indices]
    # 对top5情绪的概率进行归一化
    total_prob = sum(prob for _, prob in avg_emotions)
    avg_emotions = [(emotion, prob/total_prob) for emotion, prob in avg_emotions]
    print(f"Top 5平均情绪: {avg_emotions}")  # 调试信息
    
    return avg_emotions

@app.route('/get_subreddit_data')
def get_subreddit_data():
    """获取subreddit数据"""
    data = load_subreddit_data()
    if data:
        # 转换情绪标签从数字到实际标签
        for subreddit in data['data']:
            for post in data['data'][subreddit]:
                post['emotion_labels'] = [emotion_labels_7[MAPPING_7[i]] for i in post['emotion_labels']]
        return jsonify(data)
    return jsonify({"error": "无法加载数据"}), 500

def clean_text(text):
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 移除特殊字符和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # 转换为小写
    text = text.lower()
    return text

def get_word_frequencies(posts, top_n=100):
    # 合并所有帖子的文本
    all_text = ' '.join([post['text'] for post in posts])
    # 清理文本
    cleaned_text = clean_text(all_text)
    # 分词
    words = cleaned_text.split()
    
    # 定义需要排除的占位词
    placeholder_words = {'name', '[name]', '[ name ]', 'name]', '[name','you','that','this','there','here','what','which','who','whom','this','these','those','my','your','his','her','its','our','their','whoever','whomever','whatever','whichever','really','thats','still','just','so','like','just','even','also','though','because','if','when','where','how','why','all','some','one','two','three','four','five','six','seven','eight','nine','ten','hes','her','they','them','his','hers','its','ours','theirs'}
    
    # 移除停用词和占位词
    words = [word for word in words 
             if word not in stop_words 
             and word.lower() not in placeholder_words 
             and len(word) > 2]
    
    # 统计词频
    word_freq = Counter(words)
    
    # 返回前N个高频词，确保返回正确的格式
    return [{'text': word, 'value': int(freq)} for word, freq in word_freq.most_common(top_n)]

@app.route('/get_word_cloud/<subreddit>')
def get_word_cloud(subreddit):
    """获取subreddit的词云数据"""
    try:
        data = load_subreddit_data()
        if not data or subreddit not in data['data']:
            return jsonify({'error': 'Subreddit not found'}), 404
        
        posts = data['data'][subreddit]
        word_frequencies = get_word_frequencies(posts)
        
        # 添加调试信息
        print(f"生成词云数据: {len(word_frequencies)} 个词")
        print(f"前10个高频词: {word_frequencies[:10]}")
        
        # 检查是否还有占位词
        placeholder_words = {'name', '[name]', '[ name ]', 'name]', '[name'}
        found_placeholders = [word for word in word_frequencies 
                            if word['text'].lower() in placeholder_words]
        if found_placeholders:
            print(f"警告：词云中仍存在占位词: {found_placeholders}")
        
        return jsonify(word_frequencies)
    except Exception as e:
        print(f"生成词云数据时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_reddit_file', methods=['POST'])
def analyze_reddit_file():
    """处理上传的Reddit数据文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'})
        
        # 读取并解析JSON文件
        file_data = json.load(file)
        result = process_reddit_file(file_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_reddit_url', methods=['POST'])
def analyze_reddit_url():
    """处理Reddit URL分析请求"""
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'success': False, 'error': '未提供URL'})
        
        # 生成任务ID
        task_id = str(int(time.time()))
        
        # 初始化任务状态
        analysis_tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'result': None,
            'error': None
        }
        
        # 创建新线程执行爬取任务
        thread = threading.Thread(
            target=crawl_reddit_post,
            args=(url, task_id)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/check_analysis_status/<task_id>')
def check_analysis_status(task_id):
    """检查分析任务状态"""
    if task_id not in analysis_tasks:
        return jsonify({'success': False, 'error': '任务不存在'})
    
    task = analysis_tasks[task_id]
    response = {
        'success': True,
        'status': task['status'],
        'progress': task['progress']
    }
    
    if task['status'] == 'completed':
        response['result'] = task['result']
    elif task['status'] == 'error':
        response['error'] = task['error']
    
    return jsonify(response)

@app.route('/cancel_analysis/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    """取消分析任务"""
    if task_id in analysis_tasks:
        analysis_tasks[task_id]['status'] = 'cancelled'
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '任务不存在'})

def process_reddit_file(file_data):
    """处理上传的Reddit数据文件"""
    try:
        # 检查是否已有情感分析结果
        has_emotions = any('emotion' in comment for comment in file_data)
        
        if not has_emotions:
            # 获取情感分析器实例
            analyzer = get_emotion_analyzer()
            
            # 处理时间格式
            for comment in file_data:
                if 'created_time' in comment:
                    # 如果时间格式只有 HH:MM，添加日期部分
                    if len(comment['created_time']) == 5:  # HH:MM 格式
                        comment['created_time'] = f"2025-05-24 {comment['created_time']}:00"
            
            # 分析情感
            analyzed_comments = analyzer.analyze_comments(file_data)
            
            # 转换为时间序列数据
            timeline_data = analyzer.create_timeline_data(analyzed_comments)
            return {
                'success': True,
                'data': timeline_data,
                'message': '文件分析完成'
            }
        else:
            # 直接使用已有的情感数据
            analyzer = get_emotion_analyzer()
            timeline_data = analyzer.create_timeline_data(file_data)
            return {
                'success': True,
                'data': timeline_data,
                'message': '使用已有情感数据'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def crawl_reddit_post(url, task_id):
    """爬取Reddit帖子"""
    try:
        # 获取爬虫实例
        crawler = RedditEmotionAnalyzer.get_crawler(use_api=True)
        analyzer = get_emotion_analyzer()
        
        # 更新任务状态
        analysis_tasks[task_id]['status'] = 'crawling'
        analysis_tasks[task_id]['progress'] = 0
        
        # 爬取评论
        comments = crawler.crawl_comments(url)
        if not comments:
            analysis_tasks[task_id]['status'] = 'error'
            analysis_tasks[task_id]['error'] = '未能获取到任何有效评论'
            return
        
        analysis_tasks[task_id]['progress'] = 50
        analysis_tasks[task_id]['status'] = 'analyzing'
        
        # 分析情感
        analyzed_comments = analyzer.analyze_comments(comments)
        if not analyzed_comments:
            analysis_tasks[task_id]['status'] = 'error'
            analysis_tasks[task_id]['error'] = '情感分析后无有效数据'
            return
        
        # 创建时间序列数据
        timeline_data = analyzer.create_timeline_data(analyzed_comments)
        
        # 更新任务状态
        analysis_tasks[task_id]['status'] = 'completed'
        analysis_tasks[task_id]['progress'] = 100
        analysis_tasks[task_id]['result'] = timeline_data
        
    except Exception as e:
        analysis_tasks[task_id]['status'] = 'error'
        analysis_tasks[task_id]['error'] = str(e)
        
def get_emotion_predictor():
    """获取情感预测器实例（单例模式）"""
    global _emotion_predictor
    if _emotion_predictor is None:
        classifier = get_emotion_classifier()
        _emotion_predictor = SimpleARIMAPredictor(classifier=classifier)
        # _emotion_predictor = EmotionPredictor()
    return _emotion_predictor

@app.route('/predict_future_emotions', methods=['POST'])
def predict_future_emotions():
    """预测未来情感分布"""
    try:
        data = request.json
        if not data or 'comments' not in data:
            return jsonify({'success': False, 'error': '未提供评论数据'})
        
        # 获取预测器实例
        predictor = get_emotion_predictor()
        
        # 准备数据 - 直接使用情感概率数据
        comments = data['comments']
        
        # 将前端时间序列数据转换为ARIMA预测器需要的格式
        processed_comments = []
        for item in comments:
            comment_data = {
                'timestamp': item['timestamp'],
                'time': item.get('time', ''),
            }
            
            # 使用全局定义的7类情感标签（确保顺序一致）
            # 当前顺序: ['joy', 'neutral', 'anger', 'sadness', 'fear', 'disgust', 'surprise']
            
            # 检查数据格式
            if 'emotion_probabilities' in item:
                # 如果有emotion_probabilities字段，直接使用
                probs = item['emotion_probabilities']
                for i, emotion in enumerate(emotion_labels_7):
                    if i < len(probs):
                        comment_data[emotion] = probs[i]
                    else:
                        comment_data[emotion] = 0.0
            else:
                # 否则从各个情感字段中提取
                for emotion in emotion_labels_7:
                    comment_data[emotion] = item.get(emotion, 0.0)
            
            processed_comments.append(comment_data)
        
        # 处理评论数据
        df = predictor.process_reddit_comments(processed_comments)
        
        if len(df) == 0:
            return jsonify({'success': False, 'error': '没有有效的评论数据'})
        
        # 准备时序数据
        sequence_data = predictor.prepare_sequence_data(df, window_size=15)
        
        # 预测未来情感
        predictions = predictor.predict_future_emotions(
            sequence_data,
            num_predictions=5,
            prediction_interval=3600
        )
        
        # 转换预测结果为前端所需格式
        timeline_data = []
        print(f"\n=== 后端预测数据处理 ===")
        print(f"情感标签顺序: {emotion_labels_7}")
        
        for i, pred in enumerate(predictions):
            # ARIMA预测器已经返回7类情感概率，直接使用
            emotion_probs = pred['emotion_probabilities']
            
            # 验证数据格式
            if len(emotion_probs) != len(emotion_labels_7):
                print(f"警告：预测{i+1}的概率数组长度不匹配: {len(emotion_probs)} vs {len(emotion_labels_7)}")
                # 补齐或截断数组
                if len(emotion_probs) < len(emotion_labels_7):
                    emotion_probs.extend([0.0] * (len(emotion_labels_7) - len(emotion_probs)))
                else:
                    emotion_probs = emotion_probs[:len(emotion_labels_7)]
            
            # 确保概率总和为1
            prob_sum = sum(emotion_probs)
            if abs(prob_sum - 1.0) > 1e-10:
                print(f"警告：预测{i+1}概率总和不为1: {prob_sum}，进行归一化")
                emotion_probs = [p / prob_sum for p in emotion_probs]
            
            # 调试信息：显示预测结果
            print(f"预测 {i+1}:")
            print(f"  时间: {pred['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  情感概率: {[f'{p:.3f}' for p in emotion_probs]}")
            print(f"  对应情感: {emotion_labels_7}")
            print(f"  概率总和: {sum(emotion_probs):.6f}")
            
            # 显示主要情感分布（调试用）
            main_emotions = []
            for j, emotion in enumerate(emotion_labels_7):
                if emotion_probs[j] > 0.05:  # 显示概率大于5%的情感
                    main_emotions.append(f"{emotion}: {emotion_probs[j]:.1%}")
            print(f"  主要情感: {', '.join(main_emotions) if main_emotions else '所有情感概率都很低'}")
            
            # 验证每个情感的概率值
            for j, (emotion, prob) in enumerate(zip(emotion_labels_7, emotion_probs)):
                print(f"  {emotion} (索引{j}): {prob:.6f}")
            
            timeline_data.append({
                'timestamp': pred['timestamp'],
                'time': pred['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'emotion_probabilities': emotion_probs,
                # 添加调试信息
                'debug_info': {
                    'emotion_labels': emotion_labels_7,
                    'prob_sum': sum(emotion_probs),
                    'main_emotion': emotion_labels_7[emotion_probs.index(max(emotion_probs))],
                    'max_prob': max(emotion_probs)
                }
            })
        
        print(f"最终返回给前端的预测数据点数量: {len(timeline_data)}")
        
        # 验证返回的数据格式
        for i, item in enumerate(timeline_data):
            probs = item['emotion_probabilities']
            if len(probs) != 7:
                print(f"错误：返回数据{i+1}的概率数组长度错误: {len(probs)}")
            if abs(sum(probs) - 1.0) > 1e-10:
                print(f"错误：返回数据{i+1}的概率总和错误: {sum(probs)}")
        
        return jsonify({
            'success': True,
            'data': timeline_data,
            'emotion_labels': emotion_labels_7  # 明确返回情感标签顺序
        })
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_history', methods=['POST'])
def reset_history_route():
    """重置历史记录"""
    reset_history()
    return jsonify({'success': True})

@app.route('/analyze_shuiyuan_file', methods=['POST'])
def analyze_shuiyuan_file():
    """处理上传的水源帖子JSON文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'})
        
        # 读取并解析JSON文件
        file_data = json.load(file)
        result = process_shuiyuan_file(file_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_shuiyuan_post', methods=['POST'])
def analyze_shuiyuan_post():
    """处理水源帖子爬取分析请求"""
    try:
        post_id = request.json.get('post_id')
        if not post_id:
            return jsonify({'success': False, 'error': '未提供帖子ID'})
        
        # 生成任务ID
        task_id = str(int(time.time()))
        
        # 初始化任务状态
        analysis_tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'result': None,
            'error': None
        }
        
        # 创建新线程执行爬取任务
        thread = threading.Thread(
            target=crawl_shuiyuan_post,
            args=(post_id, task_id)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_shuiyuan_file(file_data):
    """处理上传的水源帖子数据文件"""
    try:
        # 获取情感分析器实例
        analyzer = get_emotion_analyzer()
        
        # 将水源帖子数据转换为与Reddit类似的格式
        comments = []
        for post in file_data:
            # 检查是否有英文翻译
            text_to_analyze = post.get('body_en', '')
            if not text_to_analyze or text_to_analyze.strip() == '':
                # 如果没有英文翻译，跳过或使用中文（但效果可能不好）
                continue
            
            comment_data = {
                'id': post.get('id', ''),
                'author': post.get('author', ''),
                'body': text_to_analyze,  # 使用英文翻译进行分析
                'body_original': post.get('body', ''),  # 保存原始中文
                'score': 0,  # 水源帖子没有score概念，设为0
                'created_utc': post.get('created_utc', 0),
                'created_time': post.get('created_time', ''),
                'parent_id': '',  # 水源帖子暂不处理回复关系
                'is_submitter': False,
                'depth': 0,
                'permalink': ''
            }
            comments.append(comment_data)
        
        if not comments:
            return {
                'success': False,
                'error': '没有找到可分析的英文翻译内容，请确保JSON文件包含body_en字段'
            }
        
        # 分析情感
        analyzed_comments = analyzer.analyze_comments(comments)
        
        # 转换为时间序列数据
        timeline_data = analyzer.create_timeline_data(analyzed_comments)
        
        return {
            'success': True,
            'data': timeline_data,
            'message': f'水源帖子分析完成，共分析了{len(analyzed_comments)}条发言'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def crawl_shuiyuan_post(post_id, task_id):
    """爬取水源帖子"""
    try:
        # 导入爬虫
        import sys
        sys.path.append('.')
        from post_crawler import shuiyuan_crawler
        
        # 更新任务状态
        analysis_tasks[task_id]['status'] = 'crawling'
        analysis_tasks[task_id]['progress'] = 10
        
        # 爬取帖子并生成JSON
        posts_data = shuiyuan_crawler(
            post_id=post_id, 
            to_json=True, 
            enable_translation=True,
            batch_size=10,
            convert_emojis=True
        )
        
        if not posts_data or len(posts_data) == 0:
            analysis_tasks[task_id]['status'] = 'error'
            analysis_tasks[task_id]['error'] = f'未能爬取到帖子 #{post_id} 的有效内容'
            return
        
        analysis_tasks[task_id]['progress'] = 50
        analysis_tasks[task_id]['status'] = 'analyzing'
        
        # 处理数据
        result = process_shuiyuan_file(posts_data)
        
        if not result.get('success'):
            analysis_tasks[task_id]['status'] = 'error'
            analysis_tasks[task_id]['error'] = result.get('error', '分析失败')
            return
        
        # 更新任务状态
        analysis_tasks[task_id]['status'] = 'completed'
        analysis_tasks[task_id]['progress'] = 100
        analysis_tasks[task_id]['result'] = result['data']
        
    except Exception as e:
        analysis_tasks[task_id]['status'] = 'error'
        analysis_tasks[task_id]['error'] = str(e)

if __name__ == '__main__':
    app.run(debug=True)
