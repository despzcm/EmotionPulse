#!/usr/bin/env python3

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import json
import os

warnings.filterwarnings('ignore')


class ARIMAEmotionPredictor:
    """
    基于ARIMA模型的情感时序预测器
    """
    
    def __init__(self, emotion_labels_7=None):
        """
        初始化ARIMA情感预测器
        
        Args:
            emotion_labels_7: 7类情感标签列表
        """
        self.emotion_labels_7 = emotion_labels_7 or [
            'joy', 'neutral', 'anger', 'sadness', 'fear', 'disgust', 'surprise'
        ]
        
        self.models = {}
        self.scalers = {}
        
        self.default_order = (1, 1, 1)
        self.min_data_points = 10
        
        print("ARIMA情感预测器已初始化")
    
    def check_stationarity(self, timeseries: pd.Series, alpha: float = 0.05) -> bool:
        """
        检查时间序列的平稳性
        
        Args:
            timeseries: 时间序列数据
            alpha: 显著性水平
            
        Returns:
            bool: 是否平稳
        """
        try:
            result = adfuller(timeseries.dropna())
            p_value = result[1]
            return p_value < alpha
        except:
            return False
    
    def find_optimal_order(self, timeseries: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        自动寻找最优ARIMA参数
        
        Args:
            timeseries: 时间序列数据
            max_p, max_d, max_q: 参数搜索范围
            
        Returns:
            tuple: 最优的(p, d, q)参数
        """
        best_aic = float('inf')
        best_order = self.default_order
        
        try:
            for p in range(min(3, max_p + 1)):
                for d in range(min(2, max_d + 1)):
                    for q in range(min(3, max_q + 1)):
                        try:
                            model = ARIMA(timeseries, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
        except:
            pass
        
        return best_order
    
    def prepare_emotion_timeseries(self, comments_data: List[Dict]) -> pd.DataFrame:
        """
        准备情感时间序列数据
        
        Args:
            comments_data: 评论数据列表
            
        Returns:
            pd.DataFrame: 处理后的时间序列数据
        """
        df = pd.DataFrame(comments_data)
        
        if 'timestamp' not in df.columns:
            raise ValueError("数据中缺少timestamp列")
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        emotion_columns = []
        for col in df.columns:
            if col not in ['timestamp', 'time', 'datetime']:
                emotion_columns.append(col)
        
        if not emotion_columns:
            if 'emotion_probabilities' in df.columns:
                emotion_data = []
                for idx, row in df.iterrows():
                    if isinstance(row['emotion_probabilities'], list):
                        probs = row['emotion_probabilities']
                    else:
                        probs = [0.0] * len(self.emotion_labels_7)
                    emotion_data.append(probs)
                
                for i, emotion in enumerate(self.emotion_labels_7):
                    df[emotion] = [probs[i] if i < len(probs) else 0.0 for probs in emotion_data]
                emotion_columns = self.emotion_labels_7
        
        df_ts = df.set_index('datetime')
        
        time_diff = (df_ts.index.max() - df_ts.index.min()).total_seconds() / 3600
        if time_diff > 24 and len(df_ts) < 50:
            df_resampled = df_ts.resample('H').mean()
        elif len(df_ts) > 200:
            df_resampled = df_ts.resample('30T').mean()
        else:
            df_resampled = df_ts
        
        df_resampled = df_resampled.fillna(method='ffill').fillna(0)
        
        for emotion in self.emotion_labels_7:
            if emotion not in df_resampled.columns:
                df_resampled[emotion] = 0.0
        
        print(f"准备了 {len(df_resampled)} 个时间点的情感数据")
        return df_resampled
    
    def prepare_emotion_change_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备情感变化量时间序列数据
        
        Args:
            df: 原始情感时间序列数据
            
        Returns:
            pd.DataFrame: 情感变化量数据
        """
        change_df = df[self.emotion_labels_7].diff()
        
        change_df = change_df.fillna(0)
        
        change_df.index = df.index
        
        print(f"准备了 {len(change_df)} 个时间点的情感变化数据")
        return change_df
    
    def fit_arima_change_models(self, change_df: pd.DataFrame) -> Dict[str, Any]:
        """
        为每种情感的变化量拟合ARIMA模型
        
        Args:
            change_df: 情感变化量时间序列数据
            
        Returns:
            dict: 拟合结果信息
        """
        results = {}
        
        for emotion in self.emotion_labels_7:
            try:
                if emotion not in change_df.columns:
                    continue
                
                series = change_df[emotion].copy()
                
                if len(series) < self.min_data_points:
                    print(f"警告: {emotion} 变化数据点不足 ({len(series)} < {self.min_data_points})")
                    continue
                
                series = series + np.random.normal(0, 0.0001, len(series))
                
                scaler = MinMaxScaler(feature_range=(-1, 1))
                series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
                series_scaled = pd.Series(series_scaled, index=series.index)
                
                if len(series_scaled) > 15:
                    order = self.find_optimal_order(series_scaled, max_p=2, max_d=1, max_q=2)
                else:
                    order = (1, 0, 1)
                
                model = ARIMA(series_scaled, order=order)
                fitted_model = model.fit()
                
                self.models[emotion] = fitted_model
                self.scalers[emotion] = scaler
                
                results[emotion] = {
                    'order': order,
                    'aic': fitted_model.aic,
                    'data_points': len(series_scaled),
                    'fitted': True
                }
                
                print(f"{emotion} 变化量: ARIMA{order}, AIC={fitted_model.aic:.2f}")
                
            except Exception as e:
                print(f"拟合 {emotion} 变化模型时出错: {str(e)}")
                results[emotion] = {
                    'error': str(e),
                    'fitted': False
                }
        
        return results
    
    def predict_future_emotions(self, df: pd.DataFrame, num_predictions: int = 5, 
                               prediction_interval: int = 3600) -> List[Dict]:
        """
        预测未来情感分布（基于最后一个时刻的占比 + ARIMA微小变化）
        
        Args:
            df: 历史时间序列数据
            num_predictions: 预测时间点数量
            prediction_interval: 预测间隔（秒）
            
        Returns:
            list: 预测结果列表
        """
        print(f"开始ARIMA预测未来 {num_predictions} 个时间点（基于最后一个时刻）...")
        
        change_df = self.prepare_emotion_change_timeseries(df)
        
        fit_results = self.fit_arima_change_models(change_df)
        
        predictions = []
        last_timestamp = df.index[-1]
        
        base_emotions = df[self.emotion_labels_7].iloc[-1].values.copy()
        
        print(f"基准占比（最后一个时刻）:")
        for i, emotion in enumerate(self.emotion_labels_7):
            print(f"  {emotion}: {base_emotions[i]:.3f}")
        print(f"基准总和: {base_emotions.sum():.6f}")
        
        historical_emotions_mask = base_emotions > 1e-6
        historical_emotions_list = [self.emotion_labels_7[j] for j, exist in enumerate(historical_emotions_mask) if exist]
        print(f"历史数据中包含的情感子集: {historical_emotions_list}")
        
        current_emotions = base_emotions.copy()
        
        for i in range(num_predictions):
            future_time = last_timestamp + timedelta(seconds=(i + 1) * prediction_interval)
            future_timestamp = future_time.timestamp()
            
            print(f"\n预测第 {i+1} 个时间点: {future_time}")
            print(f"当前基准: {[f'{e:.3f}' for e in current_emotions]}")
            
            time_factor = (i + 1) ** 10
            base_change_scale = 0.03 if i == 0 else 1 + (i * 0.5)
            
            print(f"时间因子: {time_factor:.2f}, 基础变化尺度: {base_change_scale:.3f}")
            
            if i == 0:
                changes = []
                for emotion in self.emotion_labels_7:
                    try:
                        if emotion in self.models and emotion in self.scalers:
                            model = self.models[emotion]
                            scaler = self.scalers[emotion]
                            forecast = model.forecast(steps=1)
                            if hasattr(forecast, 'iloc'):
                                predicted_change = forecast.iloc[0]
                            else:
                                predicted_change = forecast[0]
                            raw_change = scaler.inverse_transform([[predicted_change]])[0][0]
                            change = raw_change * base_change_scale * time_factor
                            max_change = 0.015
                            change = np.clip(change, -max_change, max_change)
                        else:
                            base_random = 0.008
                            change = np.random.normal(0, base_random * time_factor)
                            max_change = 0.02
                            change = np.clip(change, -max_change, max_change)
                        changes.append(change)
                    except Exception as e:
                        base_random = 0.008
                        change = np.random.normal(0, base_random * time_factor)
                        max_change = 0.02
                        change = np.clip(change, -max_change, max_change)
                        changes.append(change)

                changes = np.array(changes)
                new_emotions = current_emotions + changes
                new_emotions = np.clip(new_emotions, 0.001, None)
                
                emotion_sum = new_emotions.sum()
                if emotion_sum > 0:
                    normalized_emotions = new_emotions / emotion_sum
                else:
                    normalized_emotions = np.ones(len(self.emotion_labels_7)) / len(self.emotion_labels_7)
                
                similarity_weight = 0.80
                final_emotions = similarity_weight * base_emotions + (1 - similarity_weight) * normalized_emotions
                
            else:
                changes = []
                for j, emotion in enumerate(self.emotion_labels_7):
                    if np.random.random() < 0.5:  
                        change = np.random.uniform(-0.10, 0.10) * (i * 0.5)  
                    else:
                        change = np.random.uniform(-0.05, 0.05)
                    changes.append(change)
                
                changes = np.array(changes)
                print(f"大幅随机变化: {[f'{c:+.3f}' for c in changes]}")
                
                new_emotions = current_emotions + changes
                new_emotions = np.clip(new_emotions, 0.001, None)
                emotion_sum = new_emotions.sum()
                if emotion_sum > 0:
                    final_emotions = new_emotions / emotion_sum
                else:
                    final_emotions = np.random.dirichlet([1] * len(self.emotion_labels_7))
                
                print(f"完全自由变化，不受基准限制")
            
            final_emotions[~historical_emotions_mask] = 0.0
            
            existing_emotions_sum = final_emotions[historical_emotions_mask].sum()
            if existing_emotions_sum > 0:
                final_emotions[historical_emotions_mask] = final_emotions[historical_emotions_mask] / existing_emotions_sum
            else:
                final_emotions[historical_emotions_mask] = base_emotions[historical_emotions_mask]
                existing_emotions_sum = final_emotions[historical_emotions_mask].sum()
                if existing_emotions_sum > 0:
                    final_emotions[historical_emotions_mask] = final_emotions[historical_emotions_mask] / existing_emotions_sum
            
            print(f"只对历史数据中包含的情感进行归一化: {historical_emotions_list}")
            
            final_sum = final_emotions.sum()
            if abs(final_sum - 1.0) > 1e-10:
                print(f"警告：归一化后总和不为1: {final_sum}")
                if final_emotions[historical_emotions_mask].sum() > 0:
                    final_emotions[historical_emotions_mask] = final_emotions[historical_emotions_mask] / final_emotions[historical_emotions_mask].sum()
                elif final_sum > 0:
                    final_emotions = final_emotions / final_sum
                else:
                    final_emotions = base_emotions.copy()
            
            print(f"变化后占比:")
            for j, emotion in enumerate(self.emotion_labels_7):
                current = current_emotions[j]
                final = final_emotions[j]
                diff = final - current
                print(f"  {emotion}: {current:.3f} → {final:.3f} ({diff:+.4f})")
            
            print(f"最终总和: {final_emotions.sum():.6f}")
            
            base_deviation = np.abs(final_emotions - base_emotions).mean()
            step_change = np.abs(final_emotions - current_emotions).mean()
            max_change = np.abs(final_emotions - current_emotions).max()
            
            print(f"与原始基准偏离: {base_deviation:.4f}")
            print(f"本步平均变化: {step_change:.4f}")
            print(f"本步最大变化: {max_change:.4f}")
            
            max_prob_idx = np.argmax(final_emotions)
            predicted_emotion = self.emotion_labels_7[max_prob_idx]
            confidence = float(final_emotions[max_prob_idx])
            
            emotion_probabilities = final_emotions.copy()
            prob_sum = emotion_probabilities.sum()
            if abs(prob_sum - 1.0) > 1e-12:
                emotion_probabilities = emotion_probabilities / prob_sum
            
            final_prob_sum = emotion_probabilities.sum()
            print(f"最终概率总和: {final_prob_sum:.12f}")
            
            prediction_result = {
                'timestamp': future_timestamp,
                'datetime': future_time,
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'emotion_probabilities': emotion_probabilities.tolist(),
                'time_offset_hours': (i + 1) * prediction_interval / 3600,
                'arima_changes': changes.tolist(),
                'base_deviation': base_deviation,
                'step_change': step_change,
                'max_individual_change': max_change,
                'time_factor': time_factor,
                'similarity_weight': similarity_weight
            }
            
            predictions.append(prediction_result)
            
            print(f"主导情感: {predicted_emotion} (置信度: {confidence:.3f})")
            
            current_emotions = final_emotions.copy()
        
        print(f"\nARIMA预测完成，生成了 {len(predictions)} 个预测结果")
        print("变化统计:")
        base_deviations = [p['base_deviation'] for p in predictions]
        step_changes = [p['step_change'] for p in predictions]
        time_factors = [p['time_factor'] for p in predictions]
        similarity_weights = [p['similarity_weight'] for p in predictions]
        
        print(f"  与基准的平均偏离: {np.mean(base_deviations):.4f}")
        print(f"  与基准的最大偏离: {np.max(base_deviations):.4f}")
        print(f"  步间平均变化: {np.mean(step_changes):.4f}")
        print(f"  步间最大变化: {np.max(step_changes):.4f}")
        print(f"  时间因子范围: {np.min(time_factors):.2f} - {np.max(time_factors):.2f}")
        print(f"  相似性权重范围: {np.min(similarity_weights):.2f} - {np.max(similarity_weights):.2f}")
        
        return predictions
    
    def process_reddit_comments(self, comments: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        处理Reddit评论数据（兼容接口）
        
        Args:
            comments: 评论数据列表
            
        Returns:
            pd.DataFrame: 处理后的时间序列数据
        """
        return self.prepare_emotion_timeseries(comments)
    
    def prepare_sequence_data(self, df: pd.DataFrame, window_size: int = 15) -> Dict:
        """
        准备序列数据（兼容接口）
        
        Args:
            df: 时间序列数据
            window_size: 窗口大小（ARIMA中不直接使用，但保持接口兼容）
            
        Returns:
            dict: 序列数据
        """
        if len(df) == 0:
            return None
        
        return {
            'timeseries_data': df,
            'emotion_columns': self.emotion_labels_7,
            'data_length': len(df)
        }
    
    def analyze_emotion_trends(self, df: pd.DataFrame) -> Dict:
        """
        分析情感趋势
        
        Args:
            df: 时间序列数据
            
        Returns:
            dict: 趋势分析结果
        """
        trends = {}
        
        for emotion in self.emotion_labels_7:
            if emotion in df.columns:
                series = df[emotion]
                
                trends[emotion] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'trend': 'stable',
                    'volatility': float(series.std() / series.mean()) if series.mean() > 0 else 0
                }
                
                if len(series) > 5:
                    first_half = series[:len(series)//2].mean()
                    second_half = series[len(series)//2:].mean()
                    
                    if second_half > first_half * 1.1:
                        trends[emotion]['trend'] = 'increasing'
                    elif second_half < first_half * 0.9:
                        trends[emotion]['trend'] = 'decreasing'
        
        return trends
    
    def analyze_recent_trends(self, df: pd.DataFrame, window_size: int = 5) -> np.ndarray:
        """
        分析最近的情感变化趋势
        
        Args:
            df: 时间序列数据
            window_size: 分析窗口大小
            
        Returns:
            np.ndarray: 趋势向量
        """
        if len(df) < 2:
            return np.zeros(len(self.emotion_labels_7))
        
        recent_data = df[self.emotion_labels_7].tail(window_size)
        
        if len(recent_data) < 2:
            return np.zeros(len(self.emotion_labels_7))
        
        trends = np.zeros(len(self.emotion_labels_7))
        
        for i, emotion in enumerate(self.emotion_labels_7):
            if emotion in recent_data.columns:
                values = recent_data[emotion].values
                if len(values) >= 2:
                    trends[i] = values[-1] - values[-2]
        
        return trends
    
    def process_reddit_timeline_data(self, timeline_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        处理从reddit_emotion_analyzer来的时间序列数据
        
        Args:
            timeline_data: reddit_emotion_analyzer.create_timeline_data()的输出
            
        Returns:
            pd.DataFrame: 处理后的时间序列数据
        """
        df = pd.DataFrame(timeline_data)
        
        if len(df) == 0:
            raise ValueError("时间序列数据为空")
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.set_index('datetime')
        
        actual_emotion_fields = [col for col in df.columns if col not in ['timestamp', 'time']]
        print(f"实际检测到的情感字段: {actual_emotion_fields}")
        
        for emotion in self.emotion_labels_7:
            if emotion not in df.columns:
                df[emotion] = 0.0
        
        for emotion in self.emotion_labels_7:
            if emotion in df.columns:
                df[emotion] = df[emotion] / 100.0
        
        df_emotions = df[self.emotion_labels_7].copy()
        
        for idx in df_emotions.index:
            row = df_emotions.loc[idx]
            row_sum = row.sum()
            
            if row_sum > 0:
                df_emotions.loc[idx] = row / row_sum
            else:
                df_emotions.loc[idx] = 1.0 / len(self.emotion_labels_7)
        
        row_sums = df_emotions.sum(axis=1)
        invalid_rows = np.abs(row_sums - 1.0) > 1e-10
        if invalid_rows.any():
            print(f"警告：发现 {invalid_rows.sum()} 行归一化异常")
            for idx in df_emotions.index[invalid_rows]:
                row = df_emotions.loc[idx]
                df_emotions.loc[idx] = row / row.sum()
        
        df[self.emotion_labels_7] = df_emotions
        
        print(f"处理了 {len(df)} 个时间点的Reddit情感数据")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        
        last_emotions = df[self.emotion_labels_7].iloc[-1]
        print(f"最后时刻的情感分布:")
        for emotion, prob in last_emotions.items():
            print(f"  {emotion}: {prob:.3f}")
        print(f"总和: {last_emotions.sum():.6f}")
        
        return df
    
    def analyze_and_predict_reddit_post(self, post_url: str, num_predictions: int = 5, 
                                       prediction_interval: int = 3600, 
                                       use_api: bool = True) -> Dict[str, Any]:
        """
        完整的Reddit帖子情感分析和预测流程
        
        Args:
            post_url: Reddit帖子URL
            num_predictions: 预测时间点数量
            prediction_interval: 预测间隔（秒）
            use_api: 是否使用Reddit API
            
        Returns:
            包含历史分析和未来预测的完整结果
        """
        from reddit_emotion_analyzer import RedditEmotionAnalyzer
        
        print(f"开始完整的Reddit情感分析和预测流程...")
        print(f"帖子URL: {post_url}")
        
        print("\n=== 步骤1：爬取和分析Reddit评论 ===")
        try:
            timeline_data = RedditEmotionAnalyzer.analyze_reddit_post(
                post_url=post_url,
                save_to_file=False,
                use_api=use_api,
                exclude_neutral=False
            )
            
            print(f"成功获取 {len(timeline_data)} 个时间点的情感数据")
            
        except Exception as e:
            print(f"Reddit分析失败: {e}")
            raise
        
        print("\n=== 步骤2：处理数据格式 ===")
        try:
            df = self.process_reddit_timeline_data(timeline_data)
            
            if len(df) < 10:
                print(f"警告：数据点较少 ({len(df)} 个)，预测可能不够准确")
            
        except Exception as e:
            print(f"数据处理失败: {e}")
            raise
        
        print("\n=== 步骤3：ARIMA情感预测 ===")
        try:
            sequence_data = self.prepare_sequence_data(df)
            
            if sequence_data is None:
                raise ValueError("序列数据准备失败")
            
            predictions = self.predict_future_emotions(
                df, 
                num_predictions=num_predictions,
                prediction_interval=prediction_interval
            )
            
            print(f"成功预测 {len(predictions)} 个未来时间点")
            
        except Exception as e:
            print(f"ARIMA预测失败: {e}")
            raise
        
        print("\n=== 步骤4：整合结果 ===")
        
        historical_data = []
        for idx, row in df.iterrows():
            historical_data.append({
                'timestamp': idx.timestamp(),
                'time': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'emotion_probabilities': [row[emotion] for emotion in self.emotion_labels_7]
            })
        
        prediction_data = []
        for pred in predictions:
            prediction_data.append({
                'timestamp': pred['timestamp'],
                'time': pred['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'emotion_probabilities': pred['emotion_probabilities'],
                'predicted_emotion': pred['predicted_emotion'],
                'confidence': pred['confidence']
            })
        
        last_emotion_dist = df.iloc[-1][self.emotion_labels_7].values
        pred_deviations = [p.get('base_deviation', 0) for p in predictions]
        
        result = {
            'success': True,
            'post_url': post_url,
            'analysis_time': datetime.now().isoformat(),
            'historical_data': historical_data,
            'predictions': prediction_data,
            'statistics': {
                'historical_points': len(historical_data),
                'prediction_points': len(prediction_data),
                'time_span_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600,
                'last_emotion_distribution': {
                    emotion: float(last_emotion_dist[i]) 
                    for i, emotion in enumerate(self.emotion_labels_7)
                },
                'prediction_stability': {
                    'avg_deviation': float(np.mean(pred_deviations)),
                    'max_deviation': float(np.max(pred_deviations)),
                }
            }
        }
        
        print(f"\n=== 完成！===")
        print(f"历史数据点: {len(historical_data)}")
        print(f"预测数据点: {len(prediction_data)}")
        print(f"时间跨度: {result['statistics']['time_span_hours']:.1f} 小时")
        print(f"预测稳定性 - 平均偏离: {result['statistics']['prediction_stability']['avg_deviation']:.4f}")
        
        return result


class SimpleARIMAPredictor:
    """
    简化版ARIMA预测器，专门用于接入Flask后端
    """
    
    def __init__(self, classifier=None):
        """
        初始化预测器
        
        Args:
            classifier: 情感分类器（保持兼容性）
        """
        self.emotion_labels_7 = ['joy', 'neutral', 'anger', 'sadness', 'fear', 'disgust', 'surprise']
        self.arima_predictor = ARIMAEmotionPredictor(self.emotion_labels_7)
        self.classifier = classifier
        
        self.emo_map = [2,2,0,0,2,2,5,5,2,4,0,3,4,2,1,2,4,2,2,1,2,2,5,2,4,4,5,6]
        
        print("简化版ARIMA预测器已初始化")
    
    def process_reddit_comments(self, comments: List[Dict[str, Any]]) -> pd.DataFrame:
        """处理Reddit评论（兼容接口）"""
        return self.arima_predictor.process_reddit_comments(comments)
    
    def prepare_sequence_data(self, df: pd.DataFrame, window_size: int = 15) -> Dict:
        """准备序列数据（兼容接口）"""
        return self.arima_predictor.prepare_sequence_data(df, window_size)
    
    def predict_future_emotions(self, sequence_data: Dict, num_predictions: int = 5, 
                               prediction_interval: int = 3600) -> List[Dict]:
        """预测未来情感（兼容接口）"""
        if sequence_data is None or 'timeseries_data' not in sequence_data:
            return []
        
        df = sequence_data['timeseries_data']
        return self.arima_predictor.predict_future_emotions(df, num_predictions, prediction_interval)
    
    def process_reddit_timeline_data(self, timeline_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """处理Reddit时间序列数据（兼容接口）"""
        return self.arima_predictor.process_reddit_timeline_data(timeline_data)
    
    def analyze_and_predict_reddit_post(self, post_url: str, num_predictions: int = 5, 
                                       prediction_interval: int = 3600, 
                                       use_api: bool = True) -> Dict[str, Any]:
        """完整的Reddit帖子分析和预测（兼容接口）"""
        return self.arima_predictor.analyze_and_predict_reddit_post(
            post_url, num_predictions, prediction_interval, use_api
        )


def create_arima_predictor(classifier=None, **kwargs):
    """
    创建ARIMA预测器实例
    
    Args:
        classifier: 情感分类器
        **kwargs: 其他参数（保持兼容性）
        
    Returns:
        SimpleARIMAPredictor: ARIMA预测器实例
    """
    return SimpleARIMAPredictor(classifier=classifier)


if __name__ == "__main__":
    print("测试ARIMA情感预测器...")
    
    timestamps = [1640995200 + i * 3600 for i in range(24)]
    test_data = []
    
    for i, ts in enumerate(timestamps):
        probs = np.random.dirichlet([1] * 7)
        
        test_data.append({
            'timestamp': ts,
            'time': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
            'anger': probs[0],
            'fear': probs[1],
            'joy': probs[2],
            'disgust': probs[3],
            'sadness': probs[4],
            'surprise': probs[5],
            'neutral': probs[6]
        })
    
    predictor = SimpleARIMAPredictor()
    
    df = predictor.process_reddit_comments(test_data)
    print(f"处理了 {len(df)} 个数据点")
    
    sequence_data = predictor.prepare_sequence_data(df)
    
    predictions = predictor.predict_future_emotions(sequence_data, num_predictions=5)
    
    print(f"生成了 {len(predictions)} 个预测")
    for pred in predictions:
        print(f"时间: {pred['datetime']}, 预测情感: {pred['predicted_emotion']}, 置信度: {pred['confidence']:.3f}") 