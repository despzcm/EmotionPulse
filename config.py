# Reddit API配置示例
# 如果你想使用Reddit API，请将此文件重命名为config.py并填入你的凭证

REDDIT_CONFIG = {
    "client_id": "",      # 在https://www.reddit.com/prefs/apps申请
    "client_secret": "",  # 在https://www.reddit.com/prefs/apps申请
    "user_agent": "",  # 用户代理字符串
}

# 如何获取Reddit API凭证:
# 1. 访问 https://www.reddit.com/prefs/apps
# 2. 点击"Create App"或"Create Another App"
# 3. 选择"script"类型
# 4. 填写应用名称和描述
# 5. 获取client_id和client_secret 