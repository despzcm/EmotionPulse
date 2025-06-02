# -*- coding: utf-8 -*-
"""
Discourse 表情符号转换工具
将 Discourse 论坛的 :shortcode: 格式表情符号转换为 UTF8 Unicode 表情符号
"""

import re
from typing import Dict, Optional


class DiscourseEmojiConverter:
    """Discourse表情符号转换器"""
    
    def __init__(self):
        """初始化表情符号映射表"""
        self.emoji_map = self._build_emoji_map()
    
    def _build_emoji_map(self) -> Dict[str, str]:
        """
        构建Discourse shortcode到Unicode表情符号的映射表
        
        Returns:
            Dict[str, str]: shortcode到emoji的映射字典
        """
        # 基于搜索结果和常用表情符号的映射表
        emoji_map = {
            # 笑脸表情
            ":joy:": "😂",
            ":laughing:": "😆", 
            ":smiley:": "😃",
            ":smile:": "😄",
            ":grinning:": "😀",
            ":grin:": "😁",
            ":wink:": "😉",
            ":blush:": "😊",
            ":innocent:": "😇",
            ":sunglasses:": "😎",
            
            # 哭泣和伤心表情
            ":sob:": "😭",
            ":cry:": "😢",
            ":disappointed:": "😞",
            ":confused:": "😕",
            ":worried:": "😟",
            ":frowning:": "😦",
            ":anguished:": "😧",
            ":fearful:": "😨",
            ":weary:": "😩",
            ":tired_face:": "😫",
            ":disappointed_relieved:": "😥",
            ":cold_sweat:": "😰",
            ":persevere:": "😣",
            ":confounded:": "😖",
            
            # 生气和负面表情
            ":rage:": "😡",
            ":angry:": "😠",
            ":triumph:": "😤",
            ":pout:": "😡",
            ":unamused:": "😒",
            ":expressionless:": "😑",
            ":neutral_face:": "😐",
            ":no_mouth:": "😶",
            ":grimacing:": "😬",
            
            # 惊讶和其他表情
            ":astonished:": "😲",
            ":open_mouth:": "😮",
            ":hushed:": "😯",
            ":flushed:": "😳",
            ":dizzy_face:": "😵",
            ":scream:": "😱",
            ":mask:": "😷",
            ":sleeping:": "😴",
            ":sleepy:": "😪",
            ":yum:": "😋",
            ":stuck_out_tongue:": "😛",
            ":stuck_out_tongue_winking_eye:": "😜",
            ":stuck_out_tongue_closed_eyes:": "😝",
            ":heart_eyes:": "😍",
            ":kissing_heart:": "😘",
            ":kissing:": "😗",
            ":kissing_smiling_eyes:": "😙",
            ":kissing_closed_eyes:": "😚",
            ":relieved:": "😌",
            ":smirk:": "😏",
            
            # 手势
            ":thumbsup:": "👍",
            ":thumbsdown:": "👎",
            ":ok_hand:": "👌",
            ":punch:": "👊",
            ":fist:": "✊",
            ":v:": "✌️",
            ":wave:": "👋",
            ":hand:": "✋",
            ":raised_hand:": "✋",
            ":open_hands:": "👐",
            ":point_up:": "☝️",
            ":point_down:": "👇",
            ":point_left:": "👈",
            ":point_right:": "👉",
            ":raised_hands:": "🙌",
            ":pray:": "🙏",
            ":clap:": "👏",
            ":muscle:": "💪",
            
            # 心形
            ":heart:": "❤️",
            ":broken_heart:": "💔",
            ":two_hearts:": "💕",
            ":heartpulse:": "💗",
            ":heartbeat:": "💓",
            ":sparkling_heart:": "💖",
            ":cupid:": "💘",
            ":gift_heart:": "💝",
            ":heart_decoration:": "💟",
            ":purple_heart:": "💜",
            ":yellow_heart:": "💛",
            ":green_heart:": "💚",
            ":blue_heart:": "💙",
            ":orange_heart:": "🧡",
            ":black_heart:": "🖤",
            ":white_heart:": "🤍",
            ":brown_heart:": "🤎",
            
            # 常见符号
            ":fire:": "🔥",
            ":star:": "⭐",
            ":star2:": "🌟",
            ":sparkles:": "✨",
            ":boom:": "💥",
            ":collision:": "💥",
            ":exclamation:": "❗",
            ":question:": "❓",
            ":heavy_exclamation_mark:": "❗",
            ":heavy_check_mark:": "✔️",
            ":x:": "❌",
            ":cross_mark:": "❌",
            ":o:": "⭕",
            ":100:": "💯",
            ":sos:": "🆘",
            
            # 动物
            ":dog:": "🐶",
            ":cat:": "🐱",
            ":mouse:": "🐭",
            ":hamster:": "🐹",
            ":rabbit:": "🐰",
            ":fox_face:": "🦊",
            ":bear:": "🐻",
            ":panda_face:": "🐼",
            ":koala:": "🐨",
            ":tiger:": "🐯",
            ":lion:": "🦁",
            ":cow:": "🐮",
            ":pig:": "🐷",
            ":pig_nose:": "🐽",
            ":frog:": "🐸",
            ":monkey_face:": "🐵",
            ":see_no_evil:": "🙈",
            ":hear_no_evil:": "🙉",
            ":speak_no_evil:": "🙊",
            ":chicken:": "🐔",
            ":penguin:": "🐧",
            ":bird:": "🐦",
            ":baby_chick:": "🐤",
            ":hatching_chick:": "🐣",
            ":hatched_chick:": "🐥",
            ":wolf:": "🐺",
            ":boar:": "🐗",
            ":horse:": "🐴",
            ":unicorn:": "🦄",
            ":bee:": "🐝",
            ":bug:": "🐛",
            ":butterfly:": "🦋",
            ":snail:": "🐌",
            ":ant:": "🐜",
            ":beetle:": "🐞",
            ":spider:": "🕷️",
            ":scorpion:": "🦂",
            ":crab:": "🦀",
            ":snake:": "🐍",
            ":lizard:": "🦎",
            ":turtle:": "🐢",
            ":fish:": "🐟",
            ":tropical_fish:": "🐠",
            ":blowfish:": "🐡",
            ":dolphin:": "🐬",
            ":shark:": "🦈",
            ":whale:": "🐳",
            ":whale2:": "🐋",
            ":octopus:": "🐙",
            ":shell:": "🐚",
            
            # 食物
            ":apple:": "🍎",
            ":pear:": "🍐",
            ":tangerine:": "🍊",
            ":lemon:": "🍋",
            ":banana:": "🍌",
            ":watermelon:": "🍉",
            ":grapes:": "🍇",
            ":strawberry:": "🍓",
            ":melon:": "🍈",
            ":cherries:": "🍒",
            ":peach:": "🍑",
            ":pineapple:": "🍍",
            ":tomato:": "🍅",
            ":eggplant:": "🍆",
            ":hot_pepper:": "🌶️",
            ":corn:": "🌽",
            ":sweet_potato:": "🍠",
            ":honey_pot:": "🍯",
            ":bread:": "🍞",
            ":cheese:": "🧀",
            ":meat_on_bone:": "🍖",
            ":poultry_leg:": "🍗",
            ":hamburger:": "🍔",
            ":fries:": "🍟",
            ":pizza:": "🍕",
            ":hotdog:": "🌭",
            ":taco:": "🌮",
            ":burrito:": "🌯",
            ":egg:": "🥚",
            ":ramen:": "🍜",
            ":stew:": "🍲",
            ":fish_cake:": "🍥",
            ":sushi:": "🍣",
            ":bento:": "🍱",
            ":curry:": "🍛",
            ":rice:": "🍚",
            ":rice_ball:": "🍙",
            ":rice_cracker:": "🍘",
            ":oden:": "🍢",
            ":dango:": "🍡",
            ":shaved_ice:": "🍧",
            ":ice_cream:": "🍨",
            ":icecream:": "🍦",
            ":cake:": "🍰",
            ":birthday:": "🎂",
            ":custard:": "🍮",
            ":candy:": "🍬",
            ":lollipop:": "🍭",
            ":chocolate_bar:": "🍫",
            ":popcorn:": "🍿",
            ":cookie:": "🍪",
            
            # 饮料
            ":coffee:": "☕",
            ":tea:": "🍵",
            ":wine_glass:": "🍷",
            ":cocktail:": "🍸",
            ":tropical_drink:": "🍹",
            ":beer:": "🍺",
            ":beers:": "🍻",
            ":champagne:": "🍾",
            ":sake:": "🍶",
            ":milk_glass:": "🥛",
            
            # 运动
            ":soccer:": "⚽",
            ":basketball:": "🏀",
            ":football:": "🏈",
            ":baseball:": "⚾",
            ":tennis:": "🎾",
            ":volleyball:": "🏐",
            ":rugby_football:": "🏉",
            ":8ball:": "🎱",
            ":golf:": "⛳",
            ":golfing_man:": "🏌️‍♂️",
            ":ping_pong:": "🏓",
            ":badminton:": "🏸",
            ":goal_net:": "🥅",
            ":ice_hockey:": "🏒",
            ":field_hockey:": "🏑",
            ":lacrosse:": "🥍",
            ":ski:": "🎿",
            ":snowboard:": "🏂",
            ":person_fencing:": "🤺",
            ":boxing_glove:": "🥊",
            ":martial_arts_uniform:": "🥋",
            ":rowing_man:": "🚣‍♂️",
            ":swimming_man:": "🏊‍♂️",
            ":surfing_man:": "🏄‍♂️",
            ":mountain_bicyclist:": "🚵",
            ":bicyclist:": "🚴",
            ":horse_racing:": "🏇",
            ":business_suit_levitating:": "🕴️",
            ":trophy:": "🏆",
            ":medal_sports:": "🏅",
            ":medal_military:": "🎖️",
            ":1st_place_medal:": "🥇",
            ":2nd_place_medal:": "🥈",
            ":3rd_place_medal:": "🥉",
            
            # 交通工具
            ":car:": "🚗",
            ":taxi:": "🚕",
            ":blue_car:": "🚙",
            ":bus:": "🚌",
            ":trolleybus:": "🚎",
            ":racing_car:": "🏎️",
            ":police_car:": "🚓",
            ":ambulance:": "🚑",
            ":fire_engine:": "🚒",
            ":minibus:": "🚐",
            ":truck:": "🚚",
            ":articulated_lorry:": "🚛",
            ":tractor:": "🚜",
            ":kick_scooter:": "🛴",
            ":bike:": "🚲",
            ":motor_scooter:": "🛵",
            ":motorcycle:": "🏍️",
            ":rotating_light:": "🚨",
            ":oncoming_police_car:": "🚔",
            ":oncoming_bus:": "🚍",
            ":oncoming_automobile:": "🚘",
            ":oncoming_taxi:": "🚖",
            ":railway_car:": "🚃",
            ":train2:": "🚆",
            ":train:": "🚋",
            ":metro:": "🚇",
            ":light_rail:": "🚈",
            ":station:": "🚉",
            ":tram:": "🚊",
            ":monorail:": "🚝",
            ":mountain_railway:": "🚞",
            ":suspension_railway:": "🚟",
            ":aerial_tramway:": "🚡",
            ":ship:": "🚢",
            ":speedboat:": "🚤",
            ":traffic_light:": "🚥",
            ":vertical_traffic_light:": "🚦",
            ":construction:": "🚧",
            ":anchor:": "⚓",
            ":boat:": "⛵",
            ":canoe:": "🛶",
            ":sailboat:": "⛵",
            ":motorboat:": "🛥️",
            ":ferry:": "⛴️",
            ":passenger_ship:": "🛳️",
            ":rocket:": "🚀",
            ":helicopter:": "🚁",
            ":small_airplane:": "🛩️",
            ":airplane:": "✈️",
            ":flight_departure:": "🛫",
            ":flight_arrival:": "🛬",
            
            # 活动和节日
            ":tada:": "🎉",
            ":confetti_ball:": "🎊",
            ":balloon:": "🎈",
            ":birthday:": "🎂",
            ":gift:": "🎁",
            ":dolls:": "🎎",
            ":school_satchel:": "🎒",
            ":flags:": "🎏",
            ":fireworks:": "🎆",
            ":sparkler:": "🎇",
            ":wind_chime:": "🎐",
            ":rice_scene:": "🎑",
            ":jack_o_lantern:": "🎃",
            ":ghost:": "👻",
            ":santa:": "🎅",
            ":christmas_tree:": "🎄",
            ":gift:": "🎁",
            ":bell:": "🔔",
            ":no_bell:": "🔕",
            ":tanabata_tree:": "🎋",
            ":bamboo:": "🎍",
            ":crossed_flags:": "🎌",
            
            # 自然和天气
            ":sunny:": "☀️",
            ":partly_sunny:": "⛅",
            ":cloud:": "☁️",
            ":zap:": "⚡",
            ":umbrella:": "☔",
            ":snowflake:": "❄️",
            ":snowman:": "⛄",
            ":comet:": "☄️",
            ":droplet:": "💧",
            ":ocean:": "🌊",
            ":earth_africa:": "🌍",
            ":earth_americas:": "🌎",
            ":earth_asia:": "🌏",
            ":globe_with_meridians:": "🌐",
            ":new_moon:": "🌑",
            ":waxing_crescent_moon:": "🌒",
            ":first_quarter_moon:": "🌓",
            ":waxing_gibbous_moon:": "🌔",
            ":full_moon:": "🌕",
            ":waning_gibbous_moon:": "🌖",
            ":last_quarter_moon:": "🌗",
            ":waning_crescent_moon:": "🌘",
            ":crescent_moon:": "🌙",
            ":new_moon_with_face:": "🌚",
            ":first_quarter_moon_with_face:": "🌛",
            ":last_quarter_moon_with_face:": "🌜",
            ":full_moon_with_face:": "🌝",
            ":sun_with_face:": "🌞",
            ":star2:": "🌟",
            ":stars:": "🌠",
            ":thermometer:": "🌡️",
            ":mostly_sunny:": "🌤️",
            ":barely_sunny:": "🌥️",
            ":partly_sunny_rain:": "🌦️",
            ":rain_cloud:": "🌧️",
            ":snow_cloud:": "🌨️",
            ":lightning:": "🌩️",
            ":tornado:": "🌪️",
            ":fog:": "🌫️",
            ":wind_face:": "🌬️",
            ":cyclone:": "🌀",
            ":rainbow:": "🌈",
            ":closed_umbrella:": "🌂",
            ":umbrella:": "☂️",
            ":umbrella_with_rain_drops:": "☔",
            ":umbrella_on_ground:": "⛱️",
            ":zap:": "⚡",
            ":snowflake:": "❄️",
            ":snowman:": "☃️",
            ":snowman_with_snow:": "⛄",
            ":comet:": "☄️",
            ":fire:": "🔥",
            ":droplet:": "💧",
            ":ocean:": "🌊"
        }
        
        return emoji_map
    
    def convert_shortcodes(self, text: str) -> str:
        """
        将文本中的Discourse shortcode转换为Unicode表情符号
        
        Args:
            text (str): 包含shortcode的文本
            
        Returns:
            str: 转换后的文本
        """
        if not text:
            return text
        
        # 使用正则表达式匹配 :shortcode: 格式
        def replace_shortcode(match):
            shortcode = match.group(0)  # 完整的 :shortcode:
            emoji = self.emoji_map.get(shortcode, shortcode)  # 如果找不到映射就保持原样
            return emoji
        
        # 匹配 :word: 格式的表情符号
        pattern = r':[a-zA-Z0-9_+-]+:'
        converted_text = re.sub(pattern, replace_shortcode, text)
        
        return converted_text
    
    def get_available_emojis(self) -> Dict[str, str]:
        """
        获取所有可用的表情符号映射
        
        Returns:
            Dict[str, str]: shortcode到emoji的映射字典
        """
        return self.emoji_map.copy()
    
    def add_custom_emoji(self, shortcode: str, emoji: str) -> None:
        """
        添加自定义表情符号映射
        
        Args:
            shortcode (str): shortcode格式（如 ":custom:"）
            emoji (str): 对应的Unicode表情符号
        """
        if not shortcode.startswith(':') or not shortcode.endswith(':'):
            shortcode = f":{shortcode.strip(':')}:"
        
        self.emoji_map[shortcode] = emoji
    
    def remove_emoji(self, shortcode: str) -> bool:
        """
        移除表情符号映射
        
        Args:
            shortcode (str): 要移除的shortcode
            
        Returns:
            bool: 是否成功移除
        """
        if not shortcode.startswith(':') or not shortcode.endswith(':'):
            shortcode = f":{shortcode.strip(':')}:"
        
        return self.emoji_map.pop(shortcode, None) is not None


# 创建全局转换器实例
discourse_emoji_converter = DiscourseEmojiConverter()


def convert_discourse_emojis(text: str) -> str:
    """
    转换文本中的Discourse表情符号的便捷函数
    
    Args:
        text (str): 包含shortcode的文本
        
    Returns:
        str: 转换后的文本
    """
    return discourse_emoji_converter.convert_shortcodes(text)


def add_custom_emoji_mapping(shortcode: str, emoji: str) -> None:
    """
    添加自定义表情符号映射的便捷函数
    
    Args:
        shortcode (str): shortcode格式
        emoji (str): 对应的Unicode表情符号
    """
    discourse_emoji_converter.add_custom_emoji(shortcode, emoji)


if __name__ == "__main__":
    # 测试转换功能
    test_texts = [
        "我很高兴 :joy: 这真是太好了！:tada:",
        "感谢大家的支持 :pray: :heart:",
        "今天天气真好 :sunny: 适合出去玩 :smile:",
        "这个消息让我很难过 :sob: :broken_heart:",
        "加油！:muscle: 你一定可以的 :thumbsup:",
        "哈哈哈 :laughing: 这太搞笑了 :grin:",
        "生日快乐！:birthday: :cake: :gift:",
        "晚安 :sleeping: :crescent_moon: :star:",
        ":fire: 这太厉害了！:100:",
        "谢谢 :blush: 不客气 :wink:"
    ]
    
    print("=== Discourse表情符号转换测试 ===")
    for i, text in enumerate(test_texts, 1):
        converted = convert_discourse_emojis(text)
        print(f"{i}. 原文: {text}")
        print(f"   转换: {converted}")
        print()
    
    # 测试自定义表情符号
    print("=== 自定义表情符号测试 ===")
    add_custom_emoji_mapping(":test:", "🧪")
    test_custom = "这是一个测试 :test: 表情符号"
    converted_custom = convert_discourse_emojis(test_custom)
    print(f"原文: {test_custom}")
    print(f"转换: {converted_custom}")
    
    # 显示部分可用表情符号
    print("\n=== 部分可用表情符号 ===")
    emojis = discourse_emoji_converter.get_available_emojis()
    count = 0
    for shortcode, emoji in emojis.items():
        print(f"{shortcode} → {emoji}")
        count += 1
        if count >= 20:  # 只显示前20个
            break
    print(f"...总共支持 {len(emojis)} 个表情符号") 