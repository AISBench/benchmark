import os
import sys
import json
import signal
import warnings
from collections import OrderedDict, defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional, Union
from run_log import log

class WorkspaceChange:
    def __init__(self, workspace):
        self.workspace = workspace
        self.ori_workspace = os.getcwd()

    def __enter__(self):
        self.ori_workspace = os.getcwd()
        os.chdir(self.workspace)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.ori_workspace)


def deprecated(reason=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 发出废弃警告
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}",
                category=DeprecationWarning,
                stacklevel=2  # 确保指向调用者位置
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def print_json(data:dict):
    """
    以格式化的 JSON 格式打印字典
    
    参数:
        data (dict): 要打印的字典对象
    
    功能:
        1. 验证输入是否为字典类型
        2. 支持中文字符等 Unicode 字符的显示
        3. 格式化输出层级缩进
        4. 添加键值排序确保输出一致性
    
    异常:
        当传入非字典类型时抛出 TypeError
    """
    # 类型安全检查
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict type, got {type(data).__name__}")
    
    # 转换为格式化的 JSON 字符串
    # indent: 4 空格缩进
    # ensure_ascii: False 允许显示中文等 Unicode
    # sort_keys: True 按键名排序确保输出一致性
    json_str = json.dumps(
        data, 
        indent=4, 
        ensure_ascii=False, 
        sort_keys=True
    )
    
    log(json_str)


def find_all_values(data: Dict[str, Any], target_key: str) -> List[Any]:
    """
    查找所有同名键对应的值（可能非唯一）
    
    返回:
        所有匹配键对应值的列表
    """
    results = []
    
    # 使用栈进行深度优先搜索(DFS)遍历
    stack = [data]
    
    while stack:
        current = stack.pop()
        
        if isinstance(current, dict):
            # 检查当前字典中的每个键
            for key, value in current.items():
                # 如果键匹配目标键，添加值到结果
                if key == target_key:
                    results.append(value)
                
                # 将值添加到栈中以便进一步遍历
                stack.append(value)
                
        elif isinstance(current, list):
            # 将列表中的所有元素添加到栈中
            stack.extend(reversed(current))
    
    return results


def get_max_deep(data:Dict, match_value_list:List, match_key:str, target_key: str):
    max_deep = 0
    for match_value in match_value_list:
        result = find_value_in_same_level((match_key, match_value), data, target_key=target_key)
        if result:
            max_deep = max(len(result), max_deep)
    return max_deep


def find_value_in_same_level(match_pair: tuple, data: dict, target_key:str = "") -> any:
    """
    在嵌套字典中查找匹配键值对后同一层级的特定键的值
    
    参数:
        match_pair (tuple): 要匹配的键值对元组 (key, value)
        target_key (str): 要查找的目标键
        data (dict): 任意深度的嵌套字典
    
    返回:
        any: 找到的目标值，如果未找到则返回None
        
    实现逻辑:
        1. 深度优先遍历所有可能的字典结构
        2. 当找到匹配的键值对时，搜索该层是否存在目标键
        3. 递归搜索各种嵌套结构（字典、列表、元组）
    """
    match_key, match_value = match_pair

    # 当前数据结构是字典
    if isinstance(data, dict):
        # 检查是否匹配指定键值对
        if match_key in data and data[match_key] == match_value:
            # 找到匹配后，检查当前层级是否存在目标键
            return data.get(target_key, None) if target_key else data
        
        # 递归搜索所有值（可能包含子字典）
        for value in data.values():
            result = find_value_in_same_level(match_pair, value, target_key)
            if result is not None:
                return result
    
    # 当前数据结构是列表或元组
    elif isinstance(data, (list, tuple)):
        for item in data:
            result = find_value_in_same_level(match_pair, item, target_key)
            if result is not None:
                return result
    
    return None



def have_common_same_key_values(dict1: dict, dict2: dict) -> bool:
    """
    检查两个字典是否有相同的键值对（仅针对共同存在的键）
    
    参数:
        dict1 (dict): 第一个字典
        dict2 (dict): 第二个字典
    
    返回:
        bool: 如果所有共同键对应的值都相同返回True，否则返回False
        
    示例:
        have_common_same_key_values({"a":1, "b":2}, {"c":1, "a":2}) -> False
        have_common_same_key_values({"a":1, "b":2}, {"a":1, "d":2}) -> True
    """
    # 找出两个字典共有的键
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    # 检查每个共有键对应的值是否相同
    for key in common_keys:
        if dict1[key] != dict2[key]:
            return False
    
    return True

def get_spinner_frames(style: str = "clock") -> list:
    """
    获取动画帧序列
    每个帧是两个盲文字符组成的字符串
    """
    support_style = ["clock", "pulse", "ripple"]
    if style not in support_style:
        style = "clock"
    
    if style == "clock":
        # 顺时针旋转动画 (12帧)
        return [
            '⣾⣿', '⣷⣿', '⣿⣾', '⣿⣷',
            '⣿⣯', '⣿⣟', '⣿⡿', '⣿⢿',
            '⡿⣿', '⢿⣿', '⣻⣿', '⣽⣿'
        ]
    
    elif style == "pulse":
        # 脉冲动画 (3帧)
        return [
            '⣷⣾', '⣯⣽', '⣟⣻', '⡿⢿'
        ]
    
    elif style == "ripple":
        # 波纹动画 (6帧)
        return [
            '⣾⣿', '⣵⣿', '⣫⣾', '⢟⣵', 
            '⡿⣫', '⣿⢟', '⣿⡿', '⣿⢟',
            '⡿⣫', '⢟⣵', '⣫⣾', '⣵⣿'
        ]

@deprecated(reason="屏蔽SIGINT信号会导致续推功能相关的用例发送不了SIGINT导致用例失败")
def ignore_repeated_interrupts(cleanup_callback=None):
    """设置信号处理，只允许用户按一次 Ctrl+C 来中断程序"""
    # 使用函数属性存储状态
    if not hasattr(ignore_repeated_interrupts, "interrupt_received"):
        ignore_repeated_interrupts.interrupt_received = False
    
    def handler(signum, frame):
        """自定义信号处理函数"""
        if not ignore_repeated_interrupts.interrupt_received:
            # 第一次中断
            ignore_repeated_interrupts.interrupt_received = True
            print("\n\033[1;31m收到中断信号，正在终止程序... (需要耗时释放资源，后续中断将被忽略)\033[0m")
            # 如果有清理回调函数，则执行
            if cleanup_callback is not None:
                cleanup_callback()
        else:
            # 后续中断 - 完全忽略
            print("\n\033[1;33m(中断信号已被忽略，程序正在终止中...)\033[0m")
    
    # 设置自定义信号处理
    signal.signal(signal.SIGINT, handler)
    
    # 重置状态（每次调用时重置）
    ignore_repeated_interrupts.interrupt_received = False


######################################################
## 开发测试框架下的工具函数
######################################################
# 让所有用例都变成失败用例
def change_all_target_key_to_target_value(test_case_result: dict, target_key: str, target_value: any):
    # 使用双端队列作为栈
    stack = deque()
    
    # 初始压入根节点
    stack.append(test_case_result)
    
    while stack:
        # 弹出当前节点
        current = stack.pop()
        
        # 检查是否为叶子节点（包含 success 键）
        if target_key in current:
            # 修改 success 值为 False
            current[target_key] = target_value
            # 跳过进一步探索，因为叶子节点不会有子字典
            continue
        
        # 处理非叶子节点：将所有字典类型的子节点压入栈
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)