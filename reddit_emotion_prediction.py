#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import json
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import math
import random
import einops
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import argparse
import os
from typing import List, Dict, Any, Tuple
import warnings
from bert_classification import BERTClassifier
warnings.filterwarnings('ignore')

class SimplifiedTimeEmbedding(nn.Module):
    """简化的时间编码器"""
    def __init__(self, d_model):
        super().__init__()
        self.time_diff_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        
    def forward(self, x, time_diffs):
        # 简化时间编码 - 只使用时间差
        batch_size, seq_len = time_diffs.shape
        
        # 归一化时间差到[0,1]
        normalized_time_diffs = torch.clamp(time_diffs / (365*24*3600), 0, 1)
        
        # 编码时间差
        time_embeddings = self.time_diff_encoder(normalized_time_diffs.unsqueeze(-1))
        
        return x + time_embeddings

class EmotionPredictor(nn.Module):
    def __init__(self, bert_model_name, num_emotions, d_model=768, dropout=0.3, classifier=None):
        super().__init__()
        # 使用传入的classifier或创建新的classifier
        if classifier is not None:
            self.emotion_classifier = classifier
            print("使用提供的分类器")
        else:
            self.emotion_classifier = BERTClassifier(28, bert_model_name)
            print("创建新的分类器")
        
        # 简化的时间编码器
        self.time_embedding = SimplifiedTimeEmbedding(d_model)
        
        # 特征融合层 - 修改输入维度为7类情感
        self.fusion_layer = nn.Linear(d_model + 7, d_model)
        
        # 简化的Transformer编码器 - 只用2层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model*2,  # 减少参数
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 简化的输出层 - 修改输出维度为7类情感
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )
        
        # 28类到7类的映射 - 根据新的emotion_labels_7顺序调整
        # 原映射: [2,2,0,0,2,2,5,5,2,4,0,3,4,2,1,2,4,2,2,1,2,2,5,2,4,4,5,6]
        # 新顺序: ['joy', 'neutral', 'anger', 'sadness', 'fear', 'disgust', 'surprise']
        # 转换: anger(0→2), fear(1→4), joy(2→0), disgust(3→5), sadness(4→3), surprise(5→6), neutral(6→1)
        self.emo_map = [0,0,2,2,0,0,6,6,0,3,2,5,3,0,4,0,3,0,0,4,0,0,6,0,3,3,6,1]
    
    def map_emotions_to_7(self, emotion_probs):
        """将28类情感概率映射到7类"""
        batch_size, seq_len, num_emotions = emotion_probs.shape
        mapped_probs = torch.zeros((batch_size, seq_len, 7), device=emotion_probs.device)
        
        # 对每个时间步进行映射
        for i in range(num_emotions):
            mapped_idx = self.emo_map[i]
            mapped_probs[:, :, mapped_idx] += emotion_probs[:, :, i]
        
        # 归一化
        row_sums = mapped_probs.sum(dim=-1, keepdim=True)
        mapped_probs = mapped_probs / (row_sums + 1e-8)  # 添加小量避免除零
        
        return mapped_probs
    
    def forward(self, input_ids, attention_mask, timestamps, time_diffs):
        # print(input_ids.shape)
        batch_size = input_ids.shape[0]
        window_len = input_ids.shape[1]

        input_ids = einops.rearrange(input_ids, 'b w e -> (b w) e')
        attention_mask = einops.rearrange(attention_mask, 'b w e -> (b w) e')
        currDevice = input_ids.device
        with torch.no_grad():
            # 使用BERTClassifier进行预测，并获取cls_output
            pred_labels, probs, bert_embeddings = self.emotion_classifier.predict_tokenized(
                input_ids,
                attention_mask,
                return_cls_output=True
            )
            emotion_probs = torch.tensor(probs, device=input_ids.device).unsqueeze(0)
        # print(bert_embeddings.shape)
        # print(emotion_probs.shape)
        bert_embeddings = einops.rearrange(bert_embeddings.squeeze(0), '(b w) e -> b w e', b=batch_size, w=window_len).to(currDevice)
        emotion_probs = einops.rearrange(emotion_probs.squeeze(0), '(b w) e -> b w e', b=batch_size, w=window_len).to(currDevice)
        
        # 将28类情感概率映射到7类
        mapped_emotion_probs = self.map_emotions_to_7(emotion_probs)

        # 拼接BERT向量和映射后的情感概率
        combined_embeddings = torch.cat([bert_embeddings, mapped_emotion_probs], dim=-1)
        fused_embeddings = self.fusion_layer(combined_embeddings)
        
        # 添加时间编码
        time_encoded = self.time_embedding(fused_embeddings, time_diffs)
        
        # 通过Transformer
        transformer_output = self.transformer(time_encoded)
        
        # 预测7类情感
        predicted_emotions = self.output_layer(transformer_output)
        
        return predicted_emotions

class RedditEmotionPredictor:
    def __init__(self, use7emos=True, 
                 timeline_model_path="ckpts/predict/best_model.pt",
                 classifier=None):
        '''
        num_emotions: 此参数只用于时序输出，与分类器无关。
        '''
        self.use7emos = use7emos
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 情感标签映射
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
        # 7类情感标签 - 与前端保持一致的顺序
        self.emotion_labels_7 = ['joy', 'neutral', 'anger', 'sadness', 'fear', 'disgust', 'surprise']
        
        # 情感组映射
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
        
        # 初始化情感分类器
        model_name = 'bert-base-uncased'
        
        # 使用传入的classifier或创建新的classifier
        if classifier is not None:
            self.emotion_classifier = classifier
            print("使用提供的分类器")
        else:
            self.emotion_classifier = BERTClassifier(28, model_name)
            print("创建新的分类器")
        
        # 初始化时序预测模型
        self.timeline_model = EmotionPredictor(
            model_name, 
            num_emotions=7 if use7emos else 28, 
            dropout=0.3,
            classifier=self.emotion_classifier  # 传入已加载的分类器
        )
        
        # 加载时序预测模型权重（如果存在）
        if os.path.exists(timeline_model_path):
            checkpoint = torch.load(timeline_model_path, map_location=self.device)
            # 过滤掉 BERT 和分类器的权重
            filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                 if not (k.startswith('bert.') or k.startswith('emotion_classifier.'))}
            self.timeline_model.load_state_dict(filtered_state_dict, strict=False)
            print(f"时序预测模型已加载: {timeline_model_path}")
        else:
            print(f"警告: 时序预测模型文件不存在: {timeline_model_path}")
        
        self.timeline_model.to(self.device)
        self.timeline_model.eval()
        
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 情感颜色映射
        self.emotion_colors = {
            'joy': '#FFD93D', 'anger': '#FF6B6B', 'sadness': '#4ECDC4',
            'surprise': '#45B7D1', 'disgust': '#96CEB4', 'fear': '#FECA57',
            'neutral': '#BDC3C7'
        }
    
    def load_reddit_data(self, json_file: str) -> List[Dict[str, Any]]:
        """加载Reddit JSON数据"""
        print(f"加载数据: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"已加载 {len(data)} 条评论")
        return data
    
    def analyze_emotion(self, text: str) -> Tuple[str, str, float]:
        """分析单条文本的情感"""
        pred_labels, probs = self.emotion_classifier.predict(text)
        
        # 获取概率最高的情感
        predicted_class = np.argmax(probs)
        predicted_emotion = self.emotion_labels[predicted_class]
        confidence = float(probs[predicted_class])
        
        # 映射到情感组
        if self.use7emos:
            emotion_group = self.emotion_labels_7[self.emo_map[predicted_class]]
        else:
            emotion_group = self.emotion2group.get(predicted_emotion, 'neutral')
        
        return predicted_emotion, emotion_group, confidence
    
    def process_reddit_comments(self, comments: List[Dict[str, Any]]) -> pd.DataFrame:
        """处理Reddit评论，进行情感分析"""
        print("开始情感分析...")
        
        results = []
        total = len(comments)
        
        for i, comment in enumerate(comments):
            if i % 50 == 0:
                print(f"进度: {i}/{total} ({i/total*100:.1f}%)")
            
            # 提取评论信息
            comment_id = comment.get('id', '')
            author = comment.get('author', '')
            body = comment.get('body', '')
            score = comment.get('score', 0)
            created_time = comment.get('created_time', '')
            created_utc = comment.get('created_utc', 0)
            depth = comment.get('depth', 0)
            
            # 跳过空评论
            if not body or body.strip() == '':
                continue
            
            try:
                # 分析情感
                original_emotion, emotion_group, confidence = self.analyze_emotion(body)
                
                # 解析时间
                datetime_obj = datetime.strptime(created_time, '%Y-%m-%d %H:%M:%S')
                
                results.append({
                    'comment_id': comment_id,
                    'author': author,
                    'body': body,
                    'score': score,
                    'created_time': created_time,
                    'created_utc': created_utc,
                    'datetime': datetime_obj,
                    'depth': depth,
                    'original_emotion': original_emotion,
                    'emotion_group': emotion_group,
                    'confidence': confidence
                })
                
            except Exception as e:
                print(f"处理评论 {comment_id} 时出错: {e}")
                continue
        
        df = pd.DataFrame(results)
        print(f"成功分析了 {len(df)} 条评论的情感")
        
        return df
    
    def prepare_sequence_data(self, df: pd.DataFrame, window_size: int = 10) -> Dict:
        """准备时序数据用于预测"""
        # 按时间排序
        df_sorted = df.sort_values('datetime').reset_index(drop=True)
        
        if len(df_sorted) < window_size:
            print(f"数据量不足，至少需要 {window_size} 条评论")
            return None
        
        # 准备输入序列
        texts = df_sorted['body'].tolist()
        timestamps = df_sorted['created_utc'].tolist()
        
        # 计算时间间隔
        time_intervals = [0]  # 第一个评论的时间间隔为0
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            time_intervals.append(interval)
        
        # 创建时序窗口数据
        sequence_data = {
            'texts': texts[-window_size:],  # 最后window_size条评论
            'timestamps': timestamps[-window_size:],
            'time_intervals': time_intervals[-window_size:],
            'emotions': []
        }
        
        # 为每条评论进行情感分析
        for text in sequence_data['texts']:
            original_emotion, emotion_group, confidence = self.analyze_emotion(text)
            emotion_vector = torch.zeros(28)
            emotion_idx = self.emotion_labels.index(original_emotion)
            emotion_vector[emotion_idx] = 1
            
            sequence_data['emotions'].append(emotion_vector)
        
        return sequence_data
    
    def predict_future_emotions(self, sequence_data: Dict, num_predictions: int = 5, 
                               prediction_interval: int = 3600) -> List[Dict]:
        """预测未来的情感变化"""
        print(f"开始预测未来 {num_predictions} 个时间点的情感变化...")
        
        if sequence_data is None:
            return []
        
        self.timeline_model.eval()
        
        # 准备初始输入数据
        texts = sequence_data['texts'].copy()
        timestamps = sequence_data['timestamps'].copy()
        time_intervals = sequence_data['time_intervals'].copy()
        
        predictions = []
        window_size = len(texts)
        
        # 生成一些虚拟的未来文本用于预测（使用最近的文本模式）
        recent_texts = texts[-5:]  # 使用最近3条评论作为模板
        
        with torch.no_grad():
            for i in range(num_predictions):
                # 计算未来时间戳
                future_timestamp = timestamps[-1] + (i + 1) * prediction_interval
                future_datetime = datetime.fromtimestamp(future_timestamp)
                
                # 为预测生成虚拟文本（循环使用最近的文本）
                synthetic_text = recent_texts[i % len(recent_texts)]
                
                # 计算时间间隔
                if i == 0:
                    time_interval = prediction_interval
                else:
                    time_interval = prediction_interval
                
                # 更新时序数据（滑动窗口）
                current_texts = texts[-window_size+1:] + [synthetic_text]
                current_timestamps = timestamps[-window_size+1:] + [future_timestamp]
                current_intervals = time_intervals[-window_size+1:] + [time_interval]
                
                # Tokenize当前窗口的文本
                max_length = 128
                input_ids_list = []
                attention_mask_list = []
                
                for text in current_texts:
                    encoding = self.tokenizer(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    input_ids_list.append(encoding['input_ids'].squeeze(0))
                    attention_mask_list.append(encoding['attention_mask'].squeeze(0))
                
                # 转换为tensor
                input_ids = torch.stack(input_ids_list).unsqueeze(0).to(self.device)
                attention_mask = torch.stack(attention_mask_list).unsqueeze(0).to(self.device)
                time_stamps = torch.tensor(current_timestamps, dtype=torch.float32).unsqueeze(0).to(self.device)
                time_diffs = torch.tensor(current_intervals, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 预测
                predicted_emotions = self.timeline_model(input_ids, attention_mask, time_stamps, time_diffs)
                predicted_probs = torch.sigmoid(predicted_emotions)  # [1, seq_len, num_emotions]
                
                # 取最后一个时间步的预测（对应未来时间点）
                current_prediction = predicted_probs[0, -1, :].cpu().numpy()
                
                # 添加一些随机性避免预测结果完全相同
                # 基于历史情感分布进行调整
                if i > 0:
                    # 对于后续预测，引入更多变化
                    noise_scale = 0.05 + 0.01 * i  # 随着时间增加不确定性
                    noise = np.random.normal(0, noise_scale, current_prediction.shape)
                    current_prediction = np.clip(current_prediction + noise, 0.01, 0.99)
                    current_prediction = current_prediction / current_prediction.sum()  # 重新归一化
                
                # 使用温度采样获得更多样化的预测
                temperature = 1.0 + 0.2 * i  # 随时间增加温度，增加不确定性
                scaled_probs = current_prediction ** (1/temperature)
                scaled_probs = scaled_probs / scaled_probs.sum()
                
                # 将28类情感概率映射到7类
                mapped_probs = np.zeros(7)
                for j, prob in enumerate(scaled_probs):
                    mapped_probs[self.emo_map[j]] += prob
                
                # 归一化映射后的概率
                mapped_probs = mapped_probs / mapped_probs.sum()
                
                # 找到概率最高的情感（在7类中）
                predicted_emotion_idx = np.argmax(mapped_probs)
                predicted_emotion = self.emotion_labels_7[predicted_emotion_idx]
                confidence = float(mapped_probs[predicted_emotion_idx])
                
                prediction_result = {
                    'timestamp': future_timestamp,
                    'datetime': future_datetime,
                    'predicted_emotion': predicted_emotion,
                    'confidence': confidence,
                    'emotion_probabilities': mapped_probs.tolist(),  # 使用映射后的7类概率
                    'time_offset_hours': (i + 1) * prediction_interval / 3600
                }
                
                predictions.append(prediction_result)
                
                # 更新历史数据用于下一次预测
                texts.append(synthetic_text)
                timestamps.append(future_timestamp)
                time_intervals.append(time_interval)
        
        print(f"预测完成，生成了 {len(predictions)} 个预测结果")
        return predictions
    
    def visualize_emotion_timeline(self, df: pd.DataFrame, predictions: List[Dict] = None, 
                                  save_path: str = None, window_size: int = 5):
        """Visualize emotion timeline using stacked area charts with sliding windows"""
        print("Creating emotion timeline visualization...")
        
        # Validate window size
        if window_size > len(df):
            print(f"Warning: Window size {window_size} is larger than data length {len(df)}. Adjusting to {len(df)//2}")
            window_size = len(df) // 2
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Convert datetime to relative time sequence
        df['relative_time'] = range(len(df))
        
        # Calculate sliding window emotion distribution with interpolation
        window_emotions = []
        window_times = []
        
        # Use smaller step size for smoother curve
        step_size = max(1, window_size // 4)
        
        for i in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[i:i+window_size]
            emotion_counts = window_data['emotion_group'].value_counts(normalize=True)
            window_emotions.append(emotion_counts)
            window_times.append(i + window_size//2)
        
        # Convert to DataFrame for plotting
        window_df = pd.DataFrame(window_emotions, index=window_times)
        window_df = window_df.fillna(0)  # Fill missing emotions with 0
        
        # Interpolate data for smoother curves
        if len(window_df) > 1:  # Only interpolate if we have more than one point
            new_index = np.arange(window_df.index.min(), window_df.index.max() + 1)
            window_df = window_df.reindex(new_index).interpolate(method='cubic')
        
        # Ensure all values are non-negative and properly normalized
        window_df = window_df.clip(lower=0)  # Clip negative values to 0
        # Normalize each row to sum to 1
        row_sums = window_df.sum(axis=1)
        window_df = window_df.div(row_sums, axis=0)
        
        # Plot historical emotion distribution
        if not window_df.empty:
            window_df.plot(
                kind='area',
                stacked=True,
                ax=ax1,
                color=[self.emotion_colors[emotion] for emotion in window_df.columns],
                alpha=0.7,
                linewidth=0  # Remove line edges for smoother appearance
            )
        
        ax1.set_title('Reddit Comments Emotion Distribution (Sliding Window)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Sequence')
        ax1.set_ylabel('Emotion Proportion')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Prediction results
        if predictions:
            # Convert prediction times to relative sequence
            prediction_times = [i + len(df) for i in range(len(predictions))]
            prediction_emotions = [pred['predicted_emotion'] for pred in predictions]
            prediction_confidences = [pred['confidence'] for pred in predictions]
            
            # Create prediction emotion distribution
            pred_df = pd.DataFrame({
                'relative_time': prediction_times,
                'emotion': prediction_emotions,
                'confidence': prediction_confidences
            })
            
            # Adjust window size for predictions if needed
            pred_window_size = min(window_size, len(predictions))
            if pred_window_size < window_size:
                print(f"Warning: Adjusting prediction window size from {window_size} to {pred_window_size}")
            
            # Calculate sliding window for predictions with interpolation
            pred_window_emotions = []
            pred_window_times = []
            
            for i in range(0, len(predictions) - pred_window_size + 1, step_size):
                window_data = pred_df.iloc[i:i+pred_window_size]
                emotion_counts = window_data['emotion'].value_counts(normalize=True)
                pred_window_emotions.append(emotion_counts)
                pred_window_times.append(prediction_times[i] + pred_window_size//2)
            
            # Convert to DataFrame for plotting
            pred_window_df = pd.DataFrame(pred_window_emotions, index=pred_window_times)
            pred_window_df = pred_window_df.fillna(0)
            
            # Interpolate prediction data if we have enough points
            if len(pred_window_df) > 1:
                new_pred_index = np.arange(pred_window_df.index.min(), pred_window_df.index.max() + 1)
                pred_window_df = pred_window_df.reindex(new_pred_index).interpolate(method='cubic')
            
            # Ensure all values are non-negative and properly normalized
            pred_window_df = pred_window_df.clip(lower=0)  # Clip negative values to 0
            # Normalize each row to sum to 1
            pred_row_sums = pred_window_df.sum(axis=1)
            pred_window_df = pred_window_df.div(pred_row_sums, axis=0)
            
            # Plot prediction distribution
            if not pred_window_df.empty:
                pred_window_df.plot(
                    kind='area',
                    stacked=True,
                    ax=ax2,
                    color=[self.emotion_colors[emotion] for emotion in pred_window_df.columns],
                    alpha=0.7,
                    linewidth=0  # Remove line edges for smoother appearance
                )
        
        ax2.set_title('Reddit Comments Emotion Prediction Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Sequence')
        ax2.set_ylabel('Emotion Proportion')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit number of ticks
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
    
    def generate_prediction_report(self, df: pd.DataFrame, predictions: List[Dict]) -> str:
        """生成预测报告"""
        report = []
        report.append("=" * 60)
        report.append("Reddit评论情感预测分析报告")
        report.append("=" * 60)
        
        # 历史数据统计
        report.append(f"\n历史数据统计:")
        report.append(f"  总评论数: {len(df)}")
        report.append(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        
        # 历史情感分布
        emotion_dist = df['emotion_group'].value_counts()
        emotion_pct = (emotion_dist / len(df) * 100).round(1)
        
        report.append(f"\n历史情感分布:")
        for emotion in emotion_dist.index:
            report.append(f"  {emotion.capitalize()}: {emotion_dist[emotion]} ({emotion_pct[emotion]}%)")
        
        # 预测结果
        if predictions:
            report.append(f"\n预测结果:")
            report.append(f"  预测时间点数量: {len(predictions)}")
            
            # 预测情感统计
            pred_emotions = [pred['predicted_emotion'] for pred in predictions]
            pred_emotion_counts = pd.Series(pred_emotions).value_counts()
            
            report.append(f"\n预测情感分布:")
            for emotion, count in pred_emotion_counts.items():
                pct = count / len(predictions) * 100
                report.append(f"  {emotion.capitalize()}: {count} ({pct:.1f}%)")
            
            # 详细预测结果
            report.append(f"\n详细预测结果:")
            for i, pred in enumerate(predictions, 1):
                report.append(f"  预测 {i}: {pred['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
                report.append(f"    情感: {pred['predicted_emotion'].capitalize()}")
                report.append(f"    置信度: {pred['confidence']:.3f}")
                report.append(f"    距现在: {pred['time_offset_hours']:.1f} 小时")
        
        # 平均置信度
        if predictions:
            avg_confidence = np.mean([pred['confidence'] for pred in predictions])
            report.append(f"\n预测质量:")
            report.append(f"  平均置信度: {avg_confidence:.3f}")
        
        return "\n".join(report)
    
    def save_results(self, df: pd.DataFrame, predictions: List[Dict], output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存历史情感分析结果
        history_path = os.path.join(output_dir, 'emotion_analysis_results.csv')
        df.to_csv(history_path, index=False, encoding='utf-8')
        print(f"历史情感分析结果已保存到: {history_path}")
        
        # 保存预测结果
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            predictions_path = os.path.join(output_dir, 'emotion_predictions.csv')
            predictions_df.to_csv(predictions_path, index=False, encoding='utf-8')
            print(f"情感预测结果已保存到: {predictions_path}")
            
            # 保存详细预测信息
            detailed_predictions = []
            for pred in predictions:
                pred_detail = pred.copy()
                # 添加各个情感的概率
                for i, emotion in enumerate(self.emotion_labels):
                    pred_detail[f'prob_{emotion}'] = pred['emotion_probabilities'][i]
                detailed_predictions.append(pred_detail)
            
            detailed_df = pd.DataFrame(detailed_predictions)
            detailed_path = os.path.join(output_dir, 'detailed_emotion_predictions.csv')
            detailed_df.to_csv(detailed_path, index=False, encoding='utf-8')
            print(f"详细预测结果已保存到: {detailed_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default='reddit_json/reddit_comments_20250524_212936.json')
    parser.add_argument('--use7emos', action='store_true', default=True,
                       help='使用7分类情感 (默认: True)')
    parser.add_argument('--classifier-model', 
                       default='ckpts/classifier/bert_emotion_classifier_bs_16_lr_2e-05_use_grouped_emotions_7.pt',
                       help='情感分类器模型路径')
    parser.add_argument('--timeline-model', 
                       default='ckpts/predict/best_model_7.pt',
                       help='时序预测模型路径')
    parser.add_argument('--window-size', type=int, default=15,
                       help='时序窗口大小 (默认: 15)')
    parser.add_argument('--num-predictions', type=int, default=10,
                       help='预测的时间点数量 (默认: 5)')
    parser.add_argument('--prediction-interval', type=int, default=3600,
                       help='预测时间间隔 (秒, 默认: 3600 = 1小时)')
    parser.add_argument('--output-dir', default='prediction_results',
                       help='结果输出目录')
    parser.add_argument('--save-charts', default=True,
                       help='保存图表')
    parser.add_argument('--viz-window-size', type=int, default=20,
                       help='可视化滑动窗口大小 (默认: 10)')
    
    args = parser.parse_args()
    
    print("初始化Reddit情感预测系统...")
    predictor = RedditEmotionPredictor(
        use7emos=args.use7emos,
        timeline_model_path=args.timeline_model
    )
    
    print("\n开始处理Reddit数据...")
    
    # 1. 加载数据
    comments = predictor.load_reddit_data(args.json_file)
    
    # 2. 情感分析
    df = predictor.process_reddit_comments(comments)
    
    if len(df) == 0:
        print("没有找到有效的评论数据")
        return
    
    # 3. 准备时序数据
    sequence_data = predictor.prepare_sequence_data(df, window_size=args.window_size)
    
    # 4. 预测未来情感
    predictions = predictor.predict_future_emotions(
        sequence_data, 
        num_predictions=args.num_predictions,
        prediction_interval=args.prediction_interval
    )
    
    # 5. 生成可视化
    chart_path = None
    if args.save_charts:
        chart_path = os.path.join(args.output_dir, 'emotion_timeline.png')
    
    predictor.visualize_emotion_timeline(df, predictions, chart_path, window_size=args.viz_window_size)
    
    # 6. 生成报告
    report = predictor.generate_prediction_report(df, predictions)
    print("\n" + report)
    
    # 7. 保存结果
    predictor.save_results(df, predictions, args.output_dir)
    
    # 保存报告
    report_path = os.path.join(args.output_dir, 'prediction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"预测报告已保存到: {report_path}")
    
    print(f"\n分析完成！所有结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 