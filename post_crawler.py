import os
import sys
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime
import calendar

# 添加shuiyuan_exporter模块到路径
sys.path.append('./shuiyuan_exporter')
sys.path.append('./translationAPI')

from shuiyuan_exporter.main import export_exec
from shuiyuan_exporter.utils import read_cookie
import re
# 导入翻译功能
try:
    from translationAPI.translator import translate_text, translate_batch
    TRANSLATION_AVAILABLE = True
    print("🌐 翻译功能已加载")
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("⚠️ 翻译功能不可用，请检查translationAPI/translator.py")

# 导入表情符号转换功能
try:
    from translationAPI.emoji_converter import convert_discourse_emojis
    EMOJI_CONVERSION_AVAILABLE = True
    print("😊 表情符号转换功能已加载")
except ImportError:
    EMOJI_CONVERSION_AVAILABLE = False
    print("⚠️ 表情符号转换功能不可用，请检查emoji_converter.py")


def clean_text_for_translation(text: str, convert_emojis: bool = True) -> str:
    """
    清理文本，移除或转换不需要翻译的内容
    
    Args:
        text: 原始文本
        convert_emojis: 是否转换表情符号（True=转换为Unicode，False=移除）
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 处理表情符号（:emoji_name:格式）
    if convert_emojis and EMOJI_CONVERSION_AVAILABLE:
        # 转换为Unicode表情符号
        text = convert_discourse_emojis(text)
    else:
        # 移除表情符号
        text = re.sub(r':[a-zA-Z0-9_+-]+:', '', text)
    
    # 移除图片标记
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 移除链接
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def process_post_text(text: str, convert_emojis: bool = True) -> str:
    """
    处理帖子文本，转换表情符号但保留其他内容
    
    Args:
        text: 原始文本
        convert_emojis: 是否转换表情符号为Unicode
        
    Returns:
        处理后的文本
    """
    if not text:
        return ""
    
    # 转换Discourse表情符号为Unicode表情符号
    if convert_emojis and EMOJI_CONVERSION_AVAILABLE:
        text = convert_discourse_emojis(text)
    
    return text


def parse_post_content(file_path: str, enable_translation: bool = False, batch_size: int = 10, convert_emojis: bool = True) -> List[Dict]:
    """
    解析帖子markdown文件内容为结构化数据
    
    参数:
        file_path (str): 帖子文件路径
        enable_translation (bool): 是否启用翻译功能
        batch_size (int): 批量翻译的批次大小
        convert_emojis: 是否转换表情符号为Unicode
        
    返回:
        List[Dict]: 包含所有发言的列表
    """
    posts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 按分隔线分割各个发言
        sections = content.split('-------------------------')
        
        total_posts = len([s for s in sections if s.strip()])
        print(f"📄 准备解析 {total_posts} 条发言...")
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
                
            lines = section.split('\n')
            if len(lines) < 2:
                continue
            
            # 解析第一行：作者 | 时间 | #编号
            header_line = lines[0].strip()
            header_match = re.match(r'^(.+?)\s*\|\s*(.+?)\s*\|\s*#(\d+)$', header_line)
            
            if not header_match:
                continue
                
            author = header_match.group(1).strip()
            timestamp_str = header_match.group(2).strip()
            current_post_id = header_match.group(3).strip()  # 当前发言的ID
            
            # 解析时间
            try:
                # 移除 "UTC" 并解析时间
                timestamp_clean = timestamp_str.replace(' UTC', '')
                dt = datetime.strptime(timestamp_clean, '%Y-%m-%d %H:%M:%S')
                
                # 格式化为要求的格式 "2025-05-30 06:59:01"
                created_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # 转换为UTC时间戳（整数）
                created_utc = int(calendar.timegm(dt.timetuple()))
                
            except:
                created_time = timestamp_str
                created_utc = 0
            
            # 提取正文内容（除第一行外的所有行）
            content_lines = lines[1:]
            text_content = '\n'.join(content_lines).strip()
            
            # 处理引用信息
            quote_ids = []
            
            # 查找所有引用并提取信息
            # 匹配格式：[quote="用户名, post:1, topic:377979, username:用户名"]内容[/quote]
            quote_pattern = r'\[quote="[^"]*post:(\d+)[^"]*"\]\s*(.*?)\s*\[/quote\]'
            quotes = re.findall(quote_pattern, text_content, re.DOTALL)
            
            # 收集引用的发言ID
            for post_id, quoted_text in quotes:
                quote_ids.append(int(post_id))
            
            # 替换text中的quote标签，保留引用标识但使用简洁格式
            def replace_quote(match):
                quoted_text = match.group(2).strip()
                # 使用简洁的 [quote] [/quote] 格式
                return f"[quote]{quoted_text}[/quote]"
            
            text_content = re.sub(quote_pattern, replace_quote, text_content)
            
            # 处理表情符号转换（如果启用）
            if convert_emojis:
                text_content = process_post_text(text_content, convert_emojis)
            
            # 准备基本数据
            post_data = {
                "author": author,
                "created_time": created_time,  # 修改字段名
                "created_utc": created_utc,    # UTC时间戳
                "body": text_content,          # 修改字段名
                "id": int(current_post_id),    # 使用header中的发言ID
                "quote": quote_ids if quote_ids else None
            }
            
            posts.append(post_data)
    
        # 批量翻译处理
        if enable_translation and TRANSLATION_AVAILABLE and posts:
            print(f"🌐 开始批量翻译 {len(posts)} 条发言...")
            
            # 准备待翻译的文本
            texts_to_translate = []
            for post in posts:
                clean_text = clean_text_for_translation(post["body"], convert_emojis)
                texts_to_translate.append(clean_text if clean_text else None)
            
            # 分批处理翻译
            batch_count = 0
            total_batches = (len(texts_to_translate) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts_to_translate), batch_size):
                batch_count += 1
                batch_texts = texts_to_translate[i:i + batch_size]
                batch_posts = posts[i:i + batch_size]
                
                print(f"🔄 正在翻译批次 {batch_count}/{total_batches} ({len(batch_texts)} 条发言)...")
                
                try:
                    # 批量翻译
                    translated_results = translate_batch(batch_texts, from_lang='zh', to_lang='en')
                    
                    # 将翻译结果分配给对应的发言
                    for j, translated_text in enumerate(translated_results):
                        post_index = i + j
                        if post_index < len(posts):
                            posts[post_index]["body_en"] = translated_text
                    
                    success_count = sum(1 for result in translated_results if result is not None)
                    print(f"✅ 批次 {batch_count} 完成: {success_count}/{len(batch_texts)} 条翻译成功")
                    
                except Exception as e:
                    print(f"❌ 批次 {batch_count} 翻译失败: {e}")
                    # 为当前批次的发言设置翻译为None
                    for j in range(len(batch_texts)):
                        post_index = i + j
                        if post_index < len(posts):
                            posts[post_index]["body_en"] = None
                
                # 批次间延迟
                if batch_count < total_batches:
                    time.sleep(1.0)  # 批次间延迟1秒
            
            # 统计翻译结果
            total_translated = sum(1 for post in posts if post.get("body_en") is not None)
            print(f"🌐 批量翻译完成: {total_translated}/{len(posts)} 条发言翻译成功")
        
        else:
            # 如果不启用翻译，为所有发言添加空的body_en字段
            if enable_translation:
                for post in posts:
                    post["body_en"] = None
    
    except Exception as e:
        print(f"解析文件时发生错误: {e}")
        return []
    
    return posts


def shuiyuan_crawler(post_id: str, to_json: bool = False, enable_translation: bool = False, batch_size: int = 10, convert_emojis: bool = True) -> Union[bool, List[Dict]]:
    """
    爬取单个帖子并可选地返回JSON格式数据
    
    参数:
        post_id (str): 帖子编号
        to_json (bool): 是否返回JSON格式数据，默认为False
        enable_translation (bool): 是否启用翻译功能
        batch_size (int): 批量翻译的批次大小
        convert_emojis: 是否转换表情符号为Unicode
        
    返回:
        Union[bool, List[Dict]]: 
        - 当to_json=False时，返回bool表示是否成功
        - 当to_json=True时，返回包含所有发言的列表
    """
    try:
        # 确保post_id是字符串格式
        post_id = str(post_id)
        
        # 移除可能的L前缀（兼容老API）
        if post_id.startswith("L"):
            post_id = post_id[1:]
            
        print(f'开始爬取帖子 #{post_id}...')
        
        # 检查是否有缓存的cookie
        cookie_string = read_cookie()
        if not cookie_string:
            print("❌ 错误：未找到缓存的cookie，请先设置cookie")
            print("💡 解决方法：运行 cd shuiyuan_exporter && python main.py 来设置cookie")
            return False if not to_json else []
        
        print(f"✅ Cookie已加载，长度: {len(cookie_string)} 字符")
            
        # 设置保存路径 - 注意：export_exec会自动创建以post_id命名的子目录
        save_dir = "./Cache/Posts"  # 不包含post_id，避免嵌套
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 实际的帖子目录（export_exec会创建这个）
        actual_post_dir = f"{save_dir}/{post_id}"
        print(f"📁 保存目录: {actual_post_dir}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用主要的导出逻辑
        try:
            export_exec(topic=post_id, save_dir=save_dir)  # 传入基础目录，不包含post_id
        except Exception as export_error:
            print(f"❌ 爬取过程中发生错误: {export_error}")
            
            # 提供具体的错误诊断
            error_str = str(export_error)
            if "'posts_count'" in error_str:
                print("🔍 诊断：无法获取帖子信息")
                print("   可能原因：")
                print("   1. 帖子ID不存在或无效")
                print("   2. Cookie已过期，需要重新设置")
                print("   3. 帖子需要特殊权限访问")
                print("   4. 网络连接问题")
                print(f"💡 建议：尝试使用一个已知存在的帖子ID，如：377979")
            elif "网络" in error_str or "连接" in error_str:
                print("🔍 诊断：网络连接问题")
                print("💡 建议：检查网络连接和代理设置")
            elif "权限" in error_str or "403" in error_str:
                print("🔍 诊断：权限不足")
                print("💡 建议：重新设置cookie或检查账号权限")
            
            return False if not to_json else []
        
        # 计算总耗时
        total_time = time.time() - start_time
        print(f"⏱️ 帖子 #{post_id} 爬取完成，总耗时: {total_time:.2f} 秒")
        print(f"📂 文件保存在: {actual_post_dir}")
        
        # 如果需要JSON格式，解析文件内容
        if to_json:
            # 查找生成的markdown文件
            md_files = list(Path(actual_post_dir).glob(f"{post_id}*.md"))
            if md_files:
                md_file = md_files[0]
                print(f"📄 解析文件: {md_file}")
                posts_data = parse_post_content(str(md_file), enable_translation, batch_size, convert_emojis)
                
                if not posts_data:
                    print("⚠️ 警告：解析到0条发言，可能markdown文件为空或格式不正确")
                    # 检查文件内容
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if len(content.strip()) == 0:
                            print("🔍 诊断：markdown文件为空")
                        else:
                            print(f"🔍 文件内容长度: {len(content)} 字符")
                            print(f"🔍 文件前100字符: {content[:100]}...")
                    except Exception as e:
                        print(f"🔍 无法读取文件内容: {e}")
                
                # 保存JSON文件
                json_file = md_file.with_suffix('.json')
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(posts_data, f, ensure_ascii=False, indent=2)
                print(f"💾 JSON数据已保存到: {json_file}")
                
                return posts_data
            else:
                print("❌ 未找到生成的markdown文件")
                print("🔍 诊断：爬取可能失败，没有生成内容文件")
                return []
        
        return True
        
    except Exception as e:
        print(f"❌ 爬取帖子 #{post_id} 时发生未预期错误: {str(e)}")
        print(f"🔍 错误类型: {type(e).__name__}")
        return False if not to_json else []


def crawl_single_post(post_id: str) -> bool:
    """
    爬取单个帖子并保存到缓存目录（保持向后兼容）
    
    参数:
        post_id (str): 帖子编号
        
    返回:
        bool: 爬取成功返回True，失败返回False
    """
    result = shuiyuan_crawler(post_id, to_json=False, enable_translation=False)
    return isinstance(result, bool) and result


def crawl_multiple_posts(post_ids: list, to_json: bool = False, enable_translation: bool = False, batch_size: int = 10, convert_emojis: bool = True) -> dict:
    """
    批量爬取多个帖子
    
    参数:
        post_ids (list): 帖子编号列表
        to_json (bool): 是否同时生成JSON格式数据
        enable_translation (bool): 是否启用翻译功能
        batch_size (int): 批量翻译的批次大小
        convert_emojis: 是否转换表情符号为Unicode
        
    返回:
        dict: 包含成功和失败帖子的字典
    """
    results = {
        'success': [],
        'failed': [],
        'data': {} if to_json else None
    }
    
    print(f"开始批量爬取 {len(post_ids)} 个帖子...")
    if enable_translation:
        print("🌐 翻译功能已启用")
    
    for post_id in post_ids:
        print(f"\n{'='*50}")
        result = shuiyuan_crawler(post_id, to_json=to_json, enable_translation=enable_translation, batch_size=batch_size, convert_emojis=convert_emojis)
        
        if to_json:
            if isinstance(result, list) and len(result) > 0:
                results['success'].append(post_id)
                results['data'][post_id] = result
            else:
                results['failed'].append(post_id)
        else:
            if result:
                results['success'].append(post_id)
            else:
                results['failed'].append(post_id)
    
    print(f"\n{'='*50}")
    print(f"批量爬取完成！")
    print(f"成功: {len(results['success'])} 个")
    print(f"失败: {len(results['failed'])} 个")
    
    if results['failed']:
        print(f"失败的帖子: {results['failed']}")
    
    return results


if __name__ == "__main__":
    # ========== 配置变量 ==========
    # 在这里设置要爬取的帖子和选项
    
    # 单个帖子爬取示例
    single_post_id = "377979"  # 设置要爬取的单个帖子ID
    # 推荐的测试帖子ID（这些是已知存在的）:
    # "377979" - 火锅帖子（已验证存在）
    # "123456" - 请替换为实际存在的帖子ID
    # "789012" - 请替换为实际存在的帖子ID
    
    enable_json = True         # 是否生成JSON格式数据
    enable_translation = True  # 是否启用翻译功能（新增）
    batch_size = 10           # 批量翻译的批次大小（新增）
    convert_emojis = True     # 是否转换表情符号为Unicode（新增）
    
    # 批量帖子爬取示例（如果要批量爬取，请取消注释并设置post_ids）
    # post_ids = ["377979", "123456", "789012"]  # 设置要批量爬取的帖子ID列表
    post_ids = None  # 设置为None表示不进行批量爬取
    
    # 是否启用交互式模式（如果为True，会提示用户输入）
    interactive_mode = False
    
    # ========== 执行逻辑 ==========
    
    if interactive_mode:
        # 交互式输入模式
        print("=== 交互式爬取模式 ===")
        while True:
            user_input = input("\n请输入帖子编号 (输入 'quit' 退出, 添加 '--json' 生成JSON, 添加 '--translate' 启用翻译): ").strip()
            if user_input.lower() == 'quit':
                break
            
            parts = user_input.split()
            if not parts:
                continue
                
            post_id = parts[0]
            to_json = '--json' in parts
            translate = '--translate' in parts
            
            if post_id:
                result = shuiyuan_crawler(post_id, to_json=to_json, enable_translation=translate, batch_size=batch_size, convert_emojis=convert_emojis)
                if to_json and isinstance(result, list):
                    print(f"\n解析到 {len(result)} 条发言")
    
    elif post_ids is not None:
        # 批量爬取模式
        print("=== 批量爬取模式 ===")
        print(f"将要爬取的帖子: {post_ids}")
        print(f"JSON模式: {'开启' if enable_json else '关闭'}")
        print(f"翻译模式: {'开启' if enable_translation else '关闭'}")
        print(f"表情符号转换: {'开启' if convert_emojis else '关闭'}")
        print(f"批次大小: {batch_size} 条发言/批次")
        
        results = crawl_multiple_posts(post_ids, to_json=enable_json, enable_translation=enable_translation, batch_size=batch_size, convert_emojis=convert_emojis)
        
        print(f"\n批量爬取完成！")
        print(f"成功: {len(results['success'])} 个")
        print(f"失败: {len(results['failed'])} 个")
        
        if enable_json and results['data']:
            total_posts = sum(len(posts) for posts in results['data'].values())
            print(f"总共解析了 {total_posts} 条发言")
    
    else:
        # 单个帖子爬取模式
        print("=== 单个帖子爬取模式 ===")
        print(f"爬取帖子: {single_post_id}")
        print(f"JSON模式: {'开启' if enable_json else '关闭'}")
        print(f"翻译模式: {'开启' if enable_translation else '关闭'}")
        print(f"表情符号转换: {'开启' if convert_emojis else '关闭'}")
        if enable_translation:
            print(f"批次大小: {batch_size} 条发言/批次")
        
        result = shuiyuan_crawler(single_post_id, to_json=enable_json, enable_translation=enable_translation, batch_size=batch_size, convert_emojis=convert_emojis)
        
        if enable_json and isinstance(result, list):
            print(f"\n✅ 爬取成功！解析到 {len(result)} 条发言")
            if enable_translation:
                translated_count = sum(1 for post in result if post.get('body_en') is not None)
                print(f"🌐 成功翻译 {translated_count} 条发言")
        elif not enable_json and result:
            print(f"\n✅ 爬取成功！")
        else:
            print(f"\n❌ 爬取失败")
    
    print("\n程序执行完成。") 