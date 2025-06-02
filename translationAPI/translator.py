# -*- coding: utf-8 -*-
"""
百度翻译API封装函数
"""

import requests
import random
import json
import time
from hashlib import md5
from typing import Optional

class BaiduTranslator:
    def __init__(self, appid: str = '', appkey: str = ''):
        """
        初始化百度翻译器
        
        Args:
            appid: 百度翻译API的APPID
            appkey: 百度翻译API的密钥
        """
        self.appid = appid
        self.appkey = appkey
        self.endpoint = 'http://api.fanyi.baidu.com'
        self.path = '/api/trans/vip/translate'
        self.url = self.endpoint + self.path
    
    def make_md5(self, s: str, encoding: str = 'utf-8') -> str:
        """生成MD5签名"""
        return md5(s.encode(encoding)).hexdigest()
    
    def translate(self, text: str, from_lang: str = 'zh', to_lang: str = 'en', 
                  max_retries: int = 3, delay: float = 1.0) -> Optional[str]:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            from_lang: 源语言代码，默认'zh'(中文)
            to_lang: 目标语言代码，默认'en'(英文)
            max_retries: 最大重试次数
            delay: 请求间隔时间(秒)
        
        Returns:
            翻译后的文本，失败返回None
        """
        if not text or not text.strip():
            return ""
        
        # 限制文本长度，百度翻译API单次最大6000字符
        if len(text) > 6000:
            print(f"⚠️ 文本长度超过6000字符，将截断处理")
            text = text[:6000]
        
        for attempt in range(max_retries):
            try:
                # 生成salt和签名
                salt = random.randint(32768, 65536)
                sign = self.make_md5(self.appid + text + str(salt) + self.appkey)
                
                # 构建请求
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                payload = {
                    'appid': self.appid, 
                    'q': text, 
                    'from': from_lang, 
                    'to': to_lang, 
                    'salt': salt, 
                    'sign': sign
                }
                
                # 发送请求
                response = requests.post(self.url, params=payload, headers=headers, timeout=10)
                result = response.json()
                
                # 检查结果
                if 'trans_result' in result:
                    # 提取翻译结果
                    translations = []
                    for item in result['trans_result']:
                        translations.append(item['dst'])
                    return '\n'.join(translations)
                
                elif 'error_code' in result:
                    error_code = result['error_code']
                    error_msg = result.get('error_msg', '未知错误')
                    print(f"❌ 百度翻译API错误 (尝试 {attempt + 1}/{max_retries}): {error_code} - {error_msg}")
                    
                    # 如果是频率限制错误，增加延时
                    if error_code == '54003':  # 访问频率受限
                        delay *= 2
                    elif error_code == '54001':  # 签名错误
                        print("🔑 签名错误，请检查APPID和密钥")
                        return None
                    elif error_code == '58001':  # 客户端IP非法
                        print("🚫 客户端IP非法，请检查API访问权限")
                        return None
                
                else:
                    print(f"⚠️ 意外的API响应格式 (尝试 {attempt + 1}/{max_retries}): {result}")
                
            except requests.exceptions.Timeout:
                print(f"⏱️ 请求超时 (尝试 {attempt + 1}/{max_retries})")
            except requests.exceptions.RequestException as e:
                print(f"🌐 网络请求错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                print(f"❌ 翻译过程中发生错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                print(f"⏳ 等待 {delay} 秒后重试...")
                time.sleep(delay)
        
        print(f"💥 翻译失败，已达到最大重试次数 ({max_retries})")
        return None
    
    def translate_chinese_to_english(self, text: str) -> Optional[str]:
        """
        中文翻译为英文的便捷方法
        
        Args:
            text: 中文文本
            
        Returns:
            英文翻译结果
        """
        return self.translate(text, from_lang='zh', to_lang='en')
    
    def translate_batch(self, texts: list, from_lang: str = 'zh', to_lang: str = 'en', 
                       max_retries: int = 3, delay: float = 1.0) -> list:
        """
        批量翻译文本
        
        Args:
            texts: 要翻译的文本列表
            from_lang: 源语言代码，默认'zh'(中文)
            to_lang: 目标语言代码，默认'en'(英文)
            max_retries: 最大重试次数
            delay: 请求间隔时间(秒)
        
        Returns:
            翻译结果列表，与输入列表一一对应
        """
        if not texts:
            return []
        
        # 过滤空文本
        non_empty_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
        if not non_empty_texts:
            return [None] * len(texts)
        
        # 使用特殊分隔符合并文本
        separator = "\n【BATCH_SEPARATOR】\n"
        combined_text = separator.join([text for _, text in non_empty_texts])
        
        # 检查总长度
        if len(combined_text) > 6000:
            print(f"⚠️ 批量文本长度 {len(combined_text)} 超过6000字符，将分割处理")
            return self._translate_batch_chunked(texts, from_lang, to_lang, max_retries, delay)
        
        # 翻译合并后的文本
        translated_combined = self.translate(combined_text, from_lang, to_lang, max_retries, delay)
        
        if not translated_combined:
            return [None] * len(texts)
        
        # 分割翻译结果
        translated_parts = translated_combined.split("\n【BATCH_SEPARATOR】\n")
        
        # 构建结果列表
        results = [None] * len(texts)
        for i, (original_index, _) in enumerate(non_empty_texts):
            if i < len(translated_parts):
                results[original_index] = translated_parts[i].strip()
        
        return results
    
    def _translate_batch_chunked(self, texts: list, from_lang: str = 'zh', to_lang: str = 'en', 
                                max_retries: int = 3, delay: float = 1.0) -> list:
        """
        分块批量翻译（当文本总长度超过6000字符时）
        """
        results = [None] * len(texts)
        current_batch = []
        current_indices = []
        current_length = 0
        separator = "\n【BATCH_SEPARATOR】\n"
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            
            text_length = len(text) + len(separator)
            
            # 如果添加当前文本会超过限制，先处理当前批次
            if current_length + text_length > 6000 and current_batch:
                batch_results = self._process_batch(current_batch, current_indices, 
                                                   from_lang, to_lang, max_retries, delay)
                for idx, result in zip(current_indices, batch_results):
                    results[idx] = result
                
                # 重置当前批次
                current_batch = []
                current_indices = []
                current_length = 0
            
            # 添加到当前批次
            current_batch.append(text)
            current_indices.append(i)
            current_length += text_length
        
        # 处理最后一个批次
        if current_batch:
            batch_results = self._process_batch(current_batch, current_indices, 
                                               from_lang, to_lang, max_retries, delay)
            for idx, result in zip(current_indices, batch_results):
                results[idx] = result
        
        return results
    
    def _process_batch(self, texts: list, indices: list, from_lang: str, to_lang: str, 
                      max_retries: int, delay: float) -> list:
        """处理单个批次的翻译"""
        separator = "\n【BATCH_SEPARATOR】\n"
        combined_text = separator.join(texts)
        
        translated_combined = self.translate(combined_text, from_lang, to_lang, max_retries, delay)
        
        if not translated_combined:
            return [None] * len(texts)
        
        translated_parts = translated_combined.split("\n【BATCH_SEPARATOR】\n")
        
        # 确保结果数量匹配
        results = []
        for i in range(len(texts)):
            if i < len(translated_parts):
                results.append(translated_parts[i].strip())
            else:
                results.append(None)
        
        return results


# 创建全局翻译器实例
translator = BaiduTranslator()


def translate_text(text: str, from_lang: str = 'zh', to_lang: str = 'en') -> Optional[str]:
    """
    翻译文本的便捷函数
    
    Args:
        text: 要翻译的文本
        from_lang: 源语言代码
        to_lang: 目标语言代码
    
    Returns:
        翻译结果
    """
    return translator.translate(text, from_lang, to_lang)


def translate_batch(texts: list, from_lang: str = 'zh', to_lang: str = 'en') -> list:
    """
    批量翻译文本的便捷函数
    
    Args:
        texts: 要翻译的文本列表
        from_lang: 源语言代码
        to_lang: 目标语言代码
    
    Returns:
        翻译结果列表
    """
    return translator.translate_batch(texts, from_lang, to_lang)


if __name__ == "__main__":
    # 测试翻译功能
    test_text = "你好，世界！这是一个测试。"
    print(f"原文: {test_text}")
    
    result = translate_text(test_text)
    if result:
        print(f"翻译: {result}")
    else:
        print("翻译失败") 