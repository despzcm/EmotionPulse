#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit情感分析器
整合Reddit爬虫和情感分析功能的完整工具
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from bert_classification import BERTClassifier
import torch

class RedditCrawler:
    """Reddit评论爬取类"""
    
    def __init__(self, use_api: bool = True):
        """
        初始化Reddit爬虫
        
        Args:
            use_api: 是否使用Reddit API
        """
        self.use_api = use_api
        self.reddit = None
        
        # 初始化Reddit API（如果可用）
        if self.use_api:
            try:
                from config import REDDIT_CONFIG
                import praw
                self.reddit = praw.Reddit(
                    client_id=REDDIT_CONFIG["client_id"],
                    client_secret=REDDIT_CONFIG["client_secret"],
                    user_agent=REDDIT_CONFIG["user_agent"],
                    check_for_async=False
                )
                print("Reddit API初始化成功")
            except Exception as e:
                print(f"Reddit API初始化失败: {e}")
                print("将使用网页爬取模式")
                self.use_api = False
    
    def crawl_comments(self, post_url: str, filter_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        爬取Reddit评论
        
        Args:
            post_url: Reddit帖子URL
            filter_config: 评论过滤配置
            
        Returns:
            评论列表
        """
        if self.use_api and self.reddit:
            return self._crawl_comments_api(post_url, filter_config)
        else:
            return self._crawl_comments_web(post_url, filter_config)
    
    def _crawl_comments_api(self, post_url: str, filter_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """使用Reddit API爬取评论"""
        import re
        
        def extract_post_id_from_url(url: str) -> str:
            pattern = r'/comments/([a-zA-Z0-9]+)/'
            match = re.search(pattern, url)
            if match:
                return match.group(1)
            else:
                pattern = r'/([a-zA-Z0-9]+)/?$'
                match = re.search(pattern, url.rstrip('/'))
                if match:
                    return match.group(1)
            raise ValueError(f"无法从URL中提取帖子ID: {url}")
        
        post_id = extract_post_id_from_url(post_url)
        submission = self.reddit.submission(id=post_id)
        
        submission.comments.replace_more(limit=None)
        
        comments = []
        
        def extract_comment_data(comment):
            return {
                'id': comment.id,
                'author': str(comment.author) if comment.author else '[deleted]',
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'created_time': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'parent_id': comment.parent_id,
                'is_submitter': comment.is_submitter,
                'depth': 0,
                'permalink': f"https://reddit.com{comment.permalink}"
            }
        
        def get_comments_recursive(comment_list, depth=0):
            for comment in comment_list:
                if hasattr(comment, 'body'):
                    comment_data = extract_comment_data(comment)
                    comment_data['depth'] = depth
                    comments.append(comment_data)
                    
                    if comment.replies:
                        get_comments_recursive(comment.replies, depth + 1)
        
        get_comments_recursive(submission.comments)
        comments.sort(key=lambda x: x['created_utc'])
        
        return self._filter_comments(comments, filter_config)
    
    def _crawl_comments_web(self, post_url: str, filter_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """使用网页爬取评论"""
        import requests
        import re
        
        if not post_url.endswith('.json'):
            json_url = post_url + '.json' if post_url.endswith('/') else post_url + '.json'
        else:
            json_url = post_url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(json_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            comments = []
            
            def extract_comment_from_json(comment_data, depth=0):
                if comment_data['kind'] == 't1':
                    comment = comment_data['data']
                    
                    comment_info = {
                        'id': comment.get('id', ''),
                        'author': comment.get('author', '[deleted]'),
                        'body': comment.get('body', ''),
                        'score': comment.get('score', 0),
                        'created_utc': comment.get('created_utc', 0),
                        'created_time': datetime.fromtimestamp(comment.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                        'parent_id': comment.get('parent_id', ''),
                        'is_submitter': comment.get('is_submitter', False),
                        'depth': depth,
                        'permalink': f"https://reddit.com{comment.get('permalink', '')}"
                    }
                    
                    comments.append(comment_info)
                    
                    if 'replies' in comment and comment['replies']:
                        if isinstance(comment['replies'], dict) and 'data' in comment['replies']:
                            children = comment['replies']['data'].get('children', [])
                            for child in children:
                                if child['kind'] != 'more':
                                    extract_comment_from_json(child, depth + 1)
            
            if len(data) > 1 and 'data' in data[1] and 'children' in data[1]['data']:
                for child in data[1]['data']['children']:
                    if child['kind'] != 'more':
                        extract_comment_from_json(child)
            
            comments.sort(key=lambda x: x['created_utc'])
            return self._filter_comments(comments, filter_config)
            
        except Exception as e:
            print(f"网页爬取失败: {e}")
            return []
    
    def _filter_comments(self, comments: List[Dict[str, Any]], filter_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """过滤评论"""
        import re
        
        if not filter_config:
            filter_config = {
                'min_words': 5,
                'filter_urls': True
            }
        
        def is_valid_comment(comment_body: str) -> bool:
            if not comment_body or comment_body.strip() == '':
                return False
            
            # 检查是否包含URL
            if filter_config.get('filter_urls', True):
                url_patterns = [
                    r'https?://[^\s]+',
                    r'www\.[^\s]+',
                    r'[^\s]+\.(com|org|net|edu|gov|io|co|cn)[^\s]*',
                    r'reddit\.com[^\s]*',
                    r'youtu\.be[^\s]*',
                    r'bit\.ly[^\s]*',
                    r'tinyurl\.com[^\s]*',
                ]
                
                for pattern in url_patterns:
                    if re.search(pattern, comment_body, re.IGNORECASE):
                        return False
            
            min_words = filter_config.get('min_words', 5)
            if min_words > 0:
                words = re.findall(r'\b\w+\b', comment_body)
                if len(words) < min_words:
                    return False
            
            return True
        
        filtered_comments = []
        for comment in comments:
            if is_valid_comment(comment.get('body', '')):
                filtered_comments.append(comment)
        
        return filtered_comments

class EmotionAnalyzer:
    """情感分析类"""
    
    def __init__(self, 
                 model_path: str = "ckpts/classifier/bert_emotion_classifier_bs_16_lr_2e-05.pt",
                 exclude_neutral: bool = True,
                 classifier: Optional[BERTClassifier] = None):
        """
        初始化情感分析器
        
        Args:
            model_path: BERT模型路径
            exclude_neutral: 是否排除中性情感
            classifier: 可选的外部分类器实例
        """
        self.model_path = model_path
        self.exclude_neutral = exclude_neutral
        
        # 使用传入的classifier或创建新的classifier
        if classifier is not None:
            self.classifier = classifier
            print("使用提供的分类器")
        else:
            # 初始化BERT模型
            self.model_name = "./ckpts/bert-base-uncased"
            self.num_labels = 28
            self.classifier = BERTClassifier(self.num_labels, self.model_name)
            self.classifier.load_model(model_path)
            print(f"情感分类器已加载: {model_path}")
        
        # 情感标签和分组
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
        self.emotion2group = {
            'admiration': 'joy',
            'amusement': 'joy',
            'anger': 'anger',
            'annoyance': 'anger',
            'approval': 'joy',
            'caring': 'joy',
            'confusion': 'surprise',
            'curiosity': 'surprise',
            'desire': 'joy',
            'disappointment': 'sadness',
            'disapproval': 'anger',
            'disgust': 'disgust',
            'embarrassment': 'sadness',
            'excitement': 'joy',
            'fear': 'fear',
            'gratitude': 'joy',
            'grief': 'sadness',
            'joy': 'joy',
            'love': 'joy',
            'nervousness': 'fear',
            'optimism': 'joy',
            'pride': 'joy',
            'realization': 'surprise',
            'relief': 'joy',
            'remorse': 'sadness',
            'sadness': 'sadness',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
    
    def analyze_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分析评论情感
        
        Args:
            comments: 评论列表
            
        Returns:
            带有情感分析结果的评论列表
        """
        analyzed_comments = []
        
        for comment in comments:
            body = comment.get('body', '')
            if not body or body.strip() == '':
                continue
            
            try:
                # 分析情感
                pred_labels, probs = self.classifier.predict(body)
                predicted_class = np.argmax(probs)
                predicted_emotion = self.emotion_labels[predicted_class]
                confidence = float(probs[predicted_class])
                emotion_group = self.emotion2group.get(predicted_emotion, 'neutral')
                
                # 如果排除中性情感且当前情感为中性，则跳过
                if self.exclude_neutral and emotion_group == 'neutral':
                    continue
                
                # 添加情感分析结果
                comment['emotion'] = {
                    'original': predicted_emotion,
                    'group': emotion_group,
                    'confidence': confidence
                }
                
                analyzed_comments.append(comment)
                
            except Exception as e:
                print(f"处理评论 {comment.get('id', '')} 时出错: {e}")
                continue
        
        return analyzed_comments
    
    def create_timeline_data(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        创建时间序列数据
        
        Args:
            comments: 评论列表
            
        Returns:
            时间序列数据
        """
        # 按时间排序
        sorted_comments = sorted(comments, key=lambda x: x['created_utc'])
        
        # 获取所有情感组
        emotion_groups = set()
        for comment in comments:
            if 'emotion' in comment:
                emotion_groups.add(comment['emotion']['group'])
        
        # 计算总时间跨度（秒）
        total_span_seconds = 0
        if len(sorted_comments) >= 2:
            total_span_seconds = sorted_comments[-1]['created_utc'] - sorted_comments[0]['created_utc']

        # 根据时间跨度动态调整窗口大小
        if total_span_seconds <= 3600:            # ≤ 1 小时
            window_size = 10
        elif total_span_seconds <= 24 * 3600:   # ≤ 1 天
            window_size = 20
        elif total_span_seconds <= 7 * 24 * 3600:  # ≤ 7 天
            window_size = 30
        else:
            window_size = 50

        windows = []

        def format_time(ts: int) -> str:
            """根据总时间跨度返回合适的时间字符串"""
            dt = datetime.fromtimestamp(ts)
            if total_span_seconds <= 24 * 3600:
                return dt.strftime('%H:%M')  # 同一天以内，显示小时:分钟
            elif total_span_seconds <= 7 * 24 * 3600:
                return dt.strftime('%m-%d %H:%M')  # 一周以内，显示月-日 时:分
            else:
                return dt.strftime('%Y-%m-%d')  # 更长时间，显示年月日

        # 使用滑动窗口计算情感分布
        for i in range(len(sorted_comments)):
            # 获取当前窗口的评论
            start_idx = max(0, i - window_size + 1)
            window_comments = sorted_comments[start_idx:i + 1]
            
            if not window_comments:
                continue

            # 统计当前窗口的情感分布
            emotion_counts = {group: 0 for group in emotion_groups}
            total_emotions = 0

            for comment in window_comments:
                if 'emotion' in comment:
                    emotion_group = comment['emotion']['group']
                    emotion_counts[emotion_group] += 1
                    total_emotions += 1

            if total_emotions > 0:
                # 使用当前评论的时间作为时间点
                current_comment = sorted_comments[i]
                ts = current_comment['created_utc']
                window_data = {
                    'timestamp': ts,
                    'time': format_time(ts)
                }

                for group in emotion_groups:
                    window_data[group] = (emotion_counts[group] / total_emotions) * 100

                windows.append(window_data)

        return windows

class RedditEmotionAnalyzer:
    _analyzer = None
    _crawler = None
    
    @classmethod
    def get_analyzer(cls, classifier=None):
        """获取分析器实例（单例模式）"""
        if cls._analyzer is None:
            cls._analyzer = cls(classifier=classifier)
        return cls._analyzer
    
    @classmethod
    def get_crawler(cls, use_api=False):
        """获取爬虫实例（单例模式）"""
        if cls._crawler is None:
            cls._crawler = RedditCrawler(use_api=use_api)
        return cls._crawler
    
    def __init__(self, classifier=None):
        """初始化情感分析器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用传入的classifier或创建新的classifier
        if classifier is not None:
            self.classifier = classifier
            print("使用提供的分类器")
        else:
            model_name = 'ckpts/bert-base-uncased'
            self.classifier = BERTClassifier(num_labels=28, model_name=model_name)
            classifier_path = "ckpts/classifier/bert_emotion_classifier_bs_16_lr_2e-05_use_grouped_emotions_28.pt"
            if os.path.exists(classifier_path):
                self.classifier.load_model(classifier_path)
                print(f"情感分类器已加载: {classifier_path}")
            else:
                print(f"警告: 情感分类器模型文件不存在: {classifier_path}")
        
        # 初始化情感分析器
        classifier_path = "ckpts/classifier/bert_emotion_classifier_bs_16_lr_2e-05_use_grouped_emotions_28.pt"
        self.emotion_analyzer = EmotionAnalyzer(
            model_path=classifier_path,
            exclude_neutral=True,
            classifier=self.classifier
        )
    
    def analyze_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分析评论情感
        
        Args:
            comments: 评论列表
            
        Returns:
            带有情感分析结果的评论列表
        """
        return self.emotion_analyzer.analyze_comments(comments)
    
    def create_timeline_data(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        创建时间序列数据
        
        Args:
            comments: 评论列表
            
        Returns:
            时间序列数据
        """
        return self.emotion_analyzer.create_timeline_data(comments)
    
    @classmethod
    def analyze_reddit_post(cls,
                          post_url: str,
                          save_to_file: bool = False,
                          output_path: Optional[str] = None,
                          use_api: bool = True,
                          exclude_neutral: bool = True,
                          filter_config: Optional[Dict] = None) -> Union[List[Dict[str, Any]], str]:
        """
        分析Reddit帖子的评论情感
        
        Args:
            post_url: Reddit帖子URL
            save_to_file: 是否保存结果到文件
            output_path: 输出文件路径（如果save_to_file为True）
            use_api: 是否使用Reddit API
            exclude_neutral: 是否排除中性情感
            filter_config: 评论过滤配置
            
        Returns:
            如果save_to_file为False，返回包含情感分析结果的评论列表
            如果save_to_file为True，返回保存的文件路径
        """
        print(f"开始分析Reddit帖子: {post_url}")
        
        # 获取爬虫和分析器实例
        crawler = cls.get_crawler(use_api=use_api)
        analyzer = cls.get_analyzer()
        
        # 爬取评论
        comments = crawler.crawl_comments(post_url, filter_config)
        if not comments:
            raise ValueError("未能获取到任何有效评论")
        
        print(f"成功爬取 {len(comments)} 条评论")
        
        # 分析情感
        analyzed_comments = analyzer.analyze_comments(comments)
        if not analyzed_comments:
            raise ValueError("情感分析后无有效数据")
        
        print(f"成功分析 {len(analyzed_comments)} 条评论的情感")
        
        # 创建时间序列数据
        timeline_data = analyzer.create_timeline_data(analyzed_comments)
        
        # 保存结果
        if save_to_file:
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"reddit_emotion_analysis_{timestamp}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, ensure_ascii=False, indent=2)
            
            print(f"分析结果已保存到: {output_path}")
            return output_path
        
        return timeline_data

def main():
    """命令行入口"""
    # 默认参数配置
    default_config = {
        'url': None,  # Reddit帖子URL
        'output': None,  # 输出文件路径
        'use_api': True,  # 是否使用Reddit API
        'include_neutral': False,  # 是否包含中性情感
        'min_words': 5,  # 最少单词数
        'filter_urls': True  # 是否过滤包含URL的评论
    }
    
    # 示例：设置参数
    config = {
        'url': 'https://reddit.com/r/example/comments/123456/example_post/',
        'output': 'analysis_results.json',
        'use_api': True,
        'include_neutral': False,
        'min_words': 5,
        'filter_urls': True
    }
    
    # 配置过滤参数
    filter_config = {
        'min_words': config['min_words'],
        'filter_urls': config['filter_urls']
    }
    
    try:
        # 运行分析
        result = RedditEmotionAnalyzer.analyze_reddit_post(
            post_url=config['url'],
            save_to_file=True,
            output_path=config['output'],
            use_api=config['use_api'],
            exclude_neutral=not config['include_neutral'],
            filter_config=filter_config
        )
        
        print(f"分析完成！结果已保存到: {result}")
        
    except Exception as e:
        print(f"分析失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 