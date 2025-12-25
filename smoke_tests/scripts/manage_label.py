import os
import sys
import yaml
import time
import copy
import argparse
from collections import defaultdict, OrderedDict, deque
from typing import Iterable, List, Dict, Any, Tuple, Optional, Union

RESET = "\033[0m"
STRIKETHROUGH = "\033[9m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

DELETE = RED + STRIKETHROUGH

LABEL_TYPE_MAP = {
    "case_group": list,
    "case_name": str,
    "case_type": str,
    "enable": str,
    "rank_size": int,
    "script": dict,
    "timeout": int
}

KEY_ORDER = [
    "case_type", 
    "case_group", 
    "case_name", 
    "enable", 
    "script", 
    "timeout", 
    "rank_size"
]

def type_converter(value, default_type=int):
    """自定义类型转换器，使用 's:' 前缀标识字符串类型"""
    # 检查是否用 s: 前缀显式标识字符串
    if value.startswith('s:'):
        return value[2:]  # 返回去除前缀后的字符串
    
    # 尝试转为default_type
    try:
        return default_type(value)
    except ValueError:
        return value  # 转换失败则保持原始字符串

def dict_converter(value):
    """处理类字典格式的转换，支持 key:value 格式"""
    try:
        parts = value.split(':', 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: '{value}' (expected key:value)")
        
        key = parts[0].strip()
        value_part = parts[1].strip()
        
        # 对值部分进行类型推断
        if value_part.startswith('"') and value_part.endswith('"'):
            value_part = value_part[1:-1]  # 去除引号
        elif value_part.startswith("'") and value_part.endswith("'"):
            value_part = value_part[1:-1]  # 去除引号
        else:
            try:
                value_part = type_converter(value_part)
            except ValueError:
                pass
        
        return (key, value_part)
    except Exception as e:
        raise e

def validate_and_convert(input_value, label):
    """
    验证和转换输入值为标签对应的期望类型
    参数:
        input_value: 需要验证和转换的值
        label: 目标标签名称（必须存在于LABEL_TYPE_MAP中）
    返回:
        转换后的值（如果验证通过）
    异常:
        ValueError: 未知标签
        TypeError: 类型不匹配或转换失败
    """
    # 获取期望类型
    expected_type = LABEL_TYPE_MAP.get(label)
    if expected_type is None:
        raise ValueError(f"未知标签 '{label}' - 未定义类型映射")
    
    # 如果值为None或已经是正确类型，直接返回
    if input_value is None:
        return input_value
    if isinstance(input_value, expected_type):
        if isinstance(input_value, list):
            return [type_converter(each) for each in input_value]
        return type_converter(input_value)
    
    # 处理特定类型转换场景
    try:
        # 字符串到列表的转换
        if expected_type is list and isinstance(input_value, str):
            return [type_converter(input_value)]
        
        # 字典到列表的转换
        if expected_type is list and isinstance(input_value, dict):
            return list(input_value.keys())
        
        # 列表到字典的转换
        if expected_type is dict and isinstance(input_value, (list, tuple, set)):
            return dict({dict_converter(item) for item in input_value})
        
        # 列表到字符串的转换
        if expected_type is str and isinstance(input_value, list):
            return ''.join(input_value)
        
        # 列表到数值的转换
        if expected_type is int and isinstance(input_value, list):
            ret_value = type_converter(''.join(input_value))
            return int(ret_value)
        
        # 尝试通用类型转换
        return expected_type(input_value)
    except Exception as e:
        # 提供更友好的错误信息
        raise TypeError(
            f"{RED}标签 '{label}' 需要 {expected_type.__name__} 类型, "
            f"但传入的是 {type(input_value).__name__}: {input_value}{RESET}"
        ) from e

def pre_processing(tags_to_add, tags_to_remove, label):
    """
    预处理添加和移除标签数据
    参数:
        tags_to_add: 要添加的标签值
        tags_to_remove: 要移除的标签值
        label: 当前处理的标签名称
    返回:
        处理后的 (tags_to_add, tags_to_remove)
    """
    # 获取标签的期望类型
    expected_type = LABEL_TYPE_MAP.get(label)
    if expected_type is None:
        # 如果标签未定义，直接返回原始值
        return tags_to_add, tags_to_remove
    
    # 确保输入值符合期望类型
    tags_to_add = validate_and_convert(tags_to_add, label)
    tags_to_remove = validate_and_convert(tags_to_remove, label)
    
    # 根据不同类型执行不同的预处理
    if expected_type is dict:
        # 字典类型的预处理
        return preprocess_dict(tags_to_add, tags_to_remove)
    
    elif expected_type is list:
        # 列表类型的预处理
        return preprocess_list(tags_to_add, tags_to_remove)
    
    else:
        # 基本类型（str, int等）的预处理
        return preprocess_scalar(tags_to_add, tags_to_remove)

def preprocess_dict(tags_to_add, tags_to_remove):
    """处理字典类型的标签添加/移除"""
    # 确保字典类型（如果为None则转为空字典）
    tags_to_add = tags_to_add or {}
    tags_to_remove = tags_to_remove or {}
    
    # 找出冲突键（同时出现在添加和删除中）
    conflict_keys = set(tags_to_add.keys()) & set(tags_to_remove.keys())
    
    # 从两个字典中移除冲突键
    for key in conflict_keys:
        if tags_to_add[key] == tags_to_remove[key]:
            tags_to_add.pop(key, None)
            tags_to_remove.pop(key, None)
    
    # 如果没有元素，设为None
    return (tags_to_add if tags_to_add else None, 
            tags_to_remove if tags_to_remove else None)

def preprocess_list(tags_to_add, tags_to_remove):
    """处理列表类型的标签添加/移除"""
    # 确保列表类型（如果为None则转为空列表）
    tags_to_add = tags_to_add or []
    tags_to_remove = tags_to_remove or []
    
    # 找出冲突元素（同时出现在添加和删除中）
    if tags_to_add and tags_to_remove:
        conflict_items = set(tags_to_add) & set(tags_to_remove)
        # 移除冲突元素
        tags_to_add = [item for item in tags_to_add if item not in conflict_items]
        tags_to_remove = [item for item in tags_to_remove if item not in conflict_items]
    
    # 列表内部去重（保持顺序）
    tags_to_add = list(dict.fromkeys(tags_to_add)) if tags_to_add else None
    tags_to_remove = list(dict.fromkeys(tags_to_remove)) if tags_to_remove else None
    
    return tags_to_add, tags_to_remove

def preprocess_scalar(tags_to_add, tags_to_remove):
    """处理基本类型（字符串/整数等）的标签添加/移除"""
    # 如果尝试添加和删除相同的值，取消操作
    if tags_to_add is not None and tags_to_remove is not None:
        if tags_to_add == tags_to_remove:
            return None, None
    
    return tags_to_add, tags_to_remove


# 策略接口
class UpdateStrategy:
    def execute(self, data, label, tags_to_add, tags_to_remove):
        """执行更新操作，返回变更信息(added_count, removed_count, changed)"""
        raise NotImplementedError

# 字典类型更新策略
class DictUpdateStrategy(UpdateStrategy):
    def execute(self, data, label, tags_to_add, tags_to_remove):
        added_count = 0
        removed_count = 0
        changed = False
        
        current_dict = data.get(label, {})
        
        # 添加操作
        if tags_to_add:
            for key, value in tags_to_add.items():
                if key not in current_dict or current_dict[key] != value:
                    current_dict[key] = value
                    added_count += 1
                    changed = True
        
        # 移除操作
        if tags_to_remove:
            for key, value in tags_to_remove.items():
                if key in current_dict and current_dict[key] == value:
                    current_dict.pop(key)
                    removed_count += 1
                    changed = True
        
        # 更新数据
        if changed:
            if not current_dict:
                del data[label]
            else:
                data[label] = current_dict
        
        return added_count, removed_count, changed

# 列表类型更新策略
class ListUpdateStrategy(UpdateStrategy):
    def execute(self, data, label, tags_to_add, tags_to_remove):
        added_count = 0
        removed_count = 0
        changed = False
        
        current_value = data.get(label)
        current_list = list(current_value) if current_value is not None else []
        
        # 添加操作
        if tags_to_add:
            for tag in tags_to_add:
                if tag not in current_list:
                    current_list.append(tag)
                    added_count += 1
                    changed = True
        
        # 移除操作
        if tags_to_remove:
            for tag in tags_to_remove:
                if tag in current_list:
                    current_list.remove(tag)
                    removed_count += 1
                    changed = True
        
        # 更新数据（保持原始类型）
        if changed:
            if not current_list:
                del data[label]
            elif isinstance(current_value, tuple):
                data[label] = tuple(current_list)
            elif isinstance(current_value, set):
                data[label] = set(current_list)
            else:
                data[label] = current_list
        
        return added_count, removed_count, changed

# 基本类型更新策略
class ScalarUpdateStrategy(UpdateStrategy):
    def execute(self, data, label, tags_to_add, tags_to_remove):
        added_count = 0
        removed_count = 0
        changed = False
        current_value = data.get(label)
        
        # 处理移除操作
        if tags_to_remove is not None:
            if current_value == tags_to_remove:
                if tags_to_add is None:
                    if label in data:
                        del data[label]
                        removed_count += 1
                        changed = True
                elif tags_to_add != tags_to_remove:
                    if label in data:
                        del data[label]
                        removed_count += 1
                        changed = True
        
        # 处理添加操作
        if tags_to_add is not None:
            if current_value != tags_to_add:
                data[label] = tags_to_add
                added_count += 1
                changed = True
        
        return added_count, removed_count, changed

# 策略工厂
class UpdateStrategyFactory:
    @staticmethod
    def create_strategy(expected_type):
        if expected_type is dict:
            return DictUpdateStrategy()
        elif expected_type is list:
            return ListUpdateStrategy()
        else:
            return ScalarUpdateStrategy()

class UpdateOperation:
    def __init__(self, path, label, tags_to_add=None, tags_to_remove=None):
        self.__call__(path, label, tags_to_add, tags_to_remove)

    def sort_dict_to_ordered(self, dct, key_order=KEY_ORDER):
        """
        创建一个新的有序字典
        
        参数:
            dct: 原始字典
            key_order: 键的顺序列表
        
        返回:
            OrderedDict: 按指定顺序排序的有序字典
        """
        # 获取键列表
        keys = list(dct.keys())
        
        # 创建有序键列表
        ordered_keys = [key for key in key_order if key in keys]
        ordered_keys.extend(key for key in keys if key not in key_order)
        
        # 创建有序字典
        return dict(OrderedDict((key, dct[key]) for key in ordered_keys))

    def update_value(self, file_path, label, tags_to_add, tags_to_remove,
                    modified_files, added_files, removed_files):
        """
        更新YAML文件中指定标签的值
        参数:
            file_path: YAML文件路径
            label: 要修改的标签名称
            tags_to_add: 要添加的标签值
            tags_to_remove: 要移除的标签值
            modified_files: 记录修改的文件集合
            added_files: 记录添加操作的文件字典
            removed_files: 记录移除操作的文件字典
        """
        # 1. 预处理标签数据
        tags_to_add, tags_to_remove = pre_processing(tags_to_add, tags_to_remove, label)
        
        # 如果两者都为None，没有操作需要执行
        if tags_to_add is None and tags_to_remove is None:
            return
        
        # 2. 读取并解析YAML文件
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
            return
        except Exception as e:
            print(f"解析YAML文件错误: {file_path} - {str(e)}")
            return
        
        orginal_data = copy.deepcopy(data)

        # 3. 获取标签的期望类型
        expected_type = LABEL_TYPE_MAP.get(label)
        if expected_type is None:
            print(f"警告: 标签 '{label}' 未定义类型映射")
            return orginal_data
        
        # 4. 创建并执行更新策略
        strategy = UpdateStrategyFactory.create_strategy(expected_type)
        added_count, removed_count, changed = strategy.execute(data, label, tags_to_add, tags_to_remove)
        
        # 5. 如果发生变更，更新文件并记录
        if changed:
            try:
                data = self.sort_dict_to_ordered(data, KEY_ORDER)
                with open(file_path, 'w') as file:
                    yaml.dump(data, file, default_flow_style=False, sort_keys=False)
                
                # 记录变更文件
                modified_files.add(file_path)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                
                # 记录添加操作
                if added_count > 0:
                    added_files[parent_dir] = data.get(label)
                
                # 记录移除操作
                if removed_count > 0:
                    removed_files[parent_dir] = data.get(label)
            
            except Exception as e:
                print(f"写入文件错误: {file_path} - {str(e)}")
            finally:
                return orginal_data


    def find_and_process_case_yml_files(self, path, label, tags_to_add=None, tags_to_remove=None):
        modified_files = set()
        added_files = {}
        removed_files = {}

        for root, dirs, files in os.walk(path):
            for file in files:
                if file == 'case.yml':
                    file_path = os.path.join(root, file)
                    orginal_data = self.update_value(file_path, label, tags_to_add, tags_to_remove, modified_files, added_files, removed_files)
        
        # 打印统计结果
        print(f"\n{MAGENTA}Total files modified:{RESET}{GREEN}{len(modified_files)}{RESET}")
        print(f"{MAGENTA}Modified Label:{RESET} {GREEN}{label}{RESET}")
        print()
        
        def print_result_dict(result: dict, tags_type: str):
            if tags_type not in ["add", "remove"]:
                return
            
            print(f"{MAGENTA}Directories with {BLUE}{tags_type}{RESET}{MAGENTA} tags and their counts: {RESET}{GREEN}{len(result) if result else 0}{RESET}")
            if result:
                origin_values = orginal_data.get(label)
                print(f"{MAGENTA}Details for the update labels {{label (val_num): vals}}:{RESET}")
                for key, value in result.items():
                    print(f"{' ' * 4} {YELLOW}{key}{RESET} "
                        f"({MAGENTA}{len(value) if isinstance(value, Iterable) else (1 if value else 0)}{RESET}): ", end="")
                    value_list = []
                    if isinstance(value, dict):
                        print(f"{CYAN if origin_values else GREEN}[{label}]: {RESET}", end="")
                        for k, v in value.items():
                            value_list.append(f"{CYAN if origin_values and (k,v) in origin_values.items() else GREEN}({k}:{v}){RESET}")
                        if origin_values:
                            value_list.extend([f"{DELETE}({k}:{v}){RESET}" for k,v in origin_values.items() if (k,v) not in value.items()])
                        print(f"{', '.join(value_list)}")

                    elif isinstance(value, list):
                        print(f"{CYAN if origin_values else GREEN}[{label}]: {RESET}", end="")
                        for val in value:
                            value_list.append(f"{CYAN if origin_values and val in origin_values else GREEN}{val}{RESET}")
                        if origin_values:
                            value_list.extend([f"{DELETE}{val}{RESET}" for val in origin_values if val not in value])
                        print(f"{', '.join(value_list)}")
                    elif not value:
                        print(f"{DELETE}[{label}]: {origin_values}{RESET}")
                    else:
                        print(f"{CYAN if origin_values else GREEN}[{label}]: {RESET}{CYAN if value == origin_values else GREEN}{value}{RESET}")
            print()

        print_result_dict(added_files, "add")
        print_result_dict(removed_files, "remove")

    def __call__(self, path, label, tags_to_add=None, tags_to_remove=None) -> None:
        self.find_and_process_case_yml_files(path, label, tags_to_add, tags_to_remove)

class ShowOperation:
    class Config(dict):
        """特殊的字典类，可通过属性访问键值"""
        def __init__(self, **kwargs):
            super(ShowOperation.Config, self).__init__(**kwargs)
            self.__dict__ = self
    
    def __init__(self, path, label, filter_case_name=None):
        """初始化配置处理器"""
        self.filter_case_name = filter_case_name  # 保存要过滤的用例
        self.case_name_set = set()
        self.init_dir_mtime(path)
        self.__call__(path, label)

    
    def init_dir_mtime(self, path):
        """
        一次性计算所有目录的最近修改时间
        """
        # 步骤1: 收集所有目录信息
        dir_map = defaultdict(list)  # 父目录 → [子目录]
        dir_files = {}                # 目录 → 直接包含的文件
        all_dirs = set()              # 所有目录路径
        
        # 首次遍历：收集目录结构信息
        for root, dirs, files in os.walk(path):
            # 记录当前目录
            all_dirs.add(root)
            
            # 记录当前目录的文件
            dir_files[root] = [os.path.join(root, f) for f in files]
            
            # 建立父子目录关系
            for d in dirs:
                child_path = os.path.join(root, d)
                dir_map[root].append(child_path)
                all_dirs.add(child_path)
        
        # 步骤2: 自底向上计算mtime (使用深度作为处理顺序)
        depth_map = defaultdict(list)
        for dir_path in all_dirs:
            depth = len(os.path.relpath(dir_path, path).split(os.sep))
            depth_map[depth].append(dir_path)
        
        # 按深度从大到小排序 (最深的先处理)
        max_depth = max(depth_map.keys()) if depth_map else 0
        self.mtime_cache = {}
        
        for depth in range(max_depth, -1, -1):
            for dir_path in depth_map[depth]:
                current_max = 0
                
                # 1. 检查当前目录下的文件
                for file_path in dir_files.get(dir_path, []):
                    try:
                        mtime = os.path.getmtime(file_path)
                        if mtime > current_max:
                            current_max = mtime
                    except OSError:
                        continue
                
                # 2. 检查子目录的mtime (已计算)
                for child in dir_map.get(dir_path, []):
                    child_mtime = self.mtime_cache.get(child, 0)
                    if child_mtime > current_max:
                        current_max = child_mtime
                
                # 3. 记录当前目录的mtime
                self.mtime_cache[dir_path] = current_max or time.time()

    
    def read_yaml_config(self, test_case_yml: str, default_config: Dict) -> Dict:
        """
        读取YAML配置文件并更新默认配置
        
        参数:
        test_case_yml: YAML文件路径
        default_config: 默认配置字典
        """
        config = default_config.copy()
        with open(test_case_yml) as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
            config.update(yaml_config)
        return config
    
    def get_test_case_config(self, test_case_yml: str) -> Dict:
        """
        获取测试用例配置
        
        参数:
        test_case_yml: 用例配置文件路径
        """
        config = self.Config(
            case_type="default",
            case_group="default",
            case_name=test_case_yml,
            script=self.Config(
                start="run.sh",
                clean="clean.sh",
            ),
            enable=True,
            metric="metric.json",
            metric_check_rule=None,
            timeout=3,
            rank_size=1,
            yml_path=test_case_yml
        )
        return self.read_yaml_config(test_case_yml, config)
    
    def check_and_convert_if_str_or_int(self, item: Any) -> str:
        if isinstance(item, int):
            return str(item)
        try:
            int(item)
            return f"'{item}'"
        except Exception as e:
            return str(item)
    
    def print_tree(self, d: Dict, indent: int = 0, prefix: str = '', label: str = "", current_path: List[str] = None) -> None:
        """
        树状打印嵌套字典结构
        
        参数:
        d: 要打印的字典
        indent: 缩进级别
        prefix: 当前行的前缀(用于绘制连接线)
        parent_key: 父字典的键
        label: 要特殊处理的标签名
        current_path: 当前路径组件列表
        """
        if current_path is None:
            current_path = []
        if indent == 0:
            print(f"{BLUE}./{RESET}")
        # 准备所有键的列表
        keys = list(d.keys())
        for i, key in enumerate(keys):
            value = d[key]
            is_last = (i == (len(keys) - 1))  # 检查是否为最后一项
            
            # 为当前层级创建连接字符
            connector = f"{BLUE}{'└─ ' if is_last else '├─ '}{RESET}"
            
            # 为下一层级设置新的前缀
            next_prefix = prefix + f"{BLUE}{'    ' if is_last else '│   '}{RESET}"

            # 更新当前路径
            new_path = current_path + [key]
            
            # 打印当前键
            print(f"{prefix}{connector}{YELLOW}{key}{RESET}", end=":")
            
            if isinstance(value, dict):
                # 字典类型 - 继续递归
                print()
                self.print_tree(value, indent=indent+1, prefix=next_prefix, label=label, current_path=new_path)
            elif isinstance(value, list):
                # 列表类型 - 打印为多行列表
                new_prefix = prefix + f"{BLUE}{'   ' if is_last else '│  '}{RESET}"
                print()
                
                for j, item_val in enumerate(value):
                    item_connector = f"{BLUE}{'└─ ' if j == len(value)-1 else '├─ '}{RESET}"
                    
                    # 获取项的颜色
                    if label == "case_group":
                        item_color = self.check_case_group_item(item_val, new_path)
                    else:
                        item_color = GREEN
                    
                    print(f"{new_prefix}{item_connector}{item_color}{self.check_and_convert_if_str_or_int(item_val)}{RESET}")
            else:
                # 其他类型
                if label == "case_name" and (info:=self.check_valid_case_name(value, key)):
                    # 无效值用红色显示
                    print(f" {RED}{self.check_and_convert_if_str_or_int(value)} [错误原因：{info}] {RESET}")
                else:
                    print(f" {GREEN}{self.check_and_convert_if_str_or_int(value)}{RESET}")


    def check_valid_case_name(self, case_name: str, parent_dir: str) -> str:
        invalid_strings = [" "]
        for invalid_str in invalid_strings:
            if invalid_str in case_name:
                return f"用例名称: {case_name} 包含非法字符：\"{invalid_str}\""

        if case_name != parent_dir:
            return f"用例名称: {case_name} 和用例目录名称: {parent_dir} 未保持一致"
        elif case_name in self.case_name_set:
            return f"用例名称: {case_name} 重复"
        else:
            self.case_name_set.add(case_name)
            return ""

    def check_case_group_item(self, item: str, current_path: List[str]) -> str:
        """
        检查case_group项是否在路径中
        
        参数:
            item: 要检查的项
            current_path: 当前路径组件列表
        
        返回:
            如果项不在路径中，返回MAGENTA颜色
            否则返回GREEN颜色
        """
        return MAGENTA if item not in current_path else GREEN
    
    def _custom_sort(self, item: Any, path_order: Dict[str, int]) -> tuple:
        """
        自定义排序函数
        
        参数:
        item: 值列表中的当前元素
        path_order: 包含(元素, 索引)的映射字典
        """
        if item in path_order:
            # 在路径列表中的元素：第一优先级 (0) + 路径中的索引位置
            return (0, path_order[item])
        else:
            # 不在路径列表中的元素：第二优先级 (1) + 元素本身（用于字母序）
            return (1, str(item))
    
    def __call__(self, path: str, label: str) -> None:
        """
        仿函数实现：查找并显示case.yml文件
        
        参数:
        path: 要搜索的根目录
        label: case.yml中要提取并显示的标签名
        """
        case_yml_info_dict = {}
        base_path = path.rstrip(os.sep)  # 规范化基础路径
        found_target = False  # 标记是否找到目标用例

        # 遍历目录树
        for root, dirs, files in os.walk(path):
            # 如果已找到目标用例且指定了用例名，提前终止遍历
            if self.filter_case_name and found_target:
                break
                
            # 按目录修改时间排序
            dirs.sort(key=lambda d: self.mtime_cache.get(os.path.join(root, d), 0), reverse=False)
            
            for file in files:
                if file == 'case.yml':
                    file_path = os.path.join(root, file)
                    
                    # 获取配置中的标签值
                    config = self.get_test_case_config(file_path)
                    
                    # 如果指定了用例名且不匹配，跳过处理
                    if self.filter_case_name and config.get("case_name") != self.filter_case_name:
                        continue
                    
                    # 标记已找到目标用例
                    if self.filter_case_name:
                        found_target = True
                    
                    # 计算相对路径
                    rel_path = os.path.relpath(root, base_path)
                    if rel_path == ".":
                        rel_path = ""
                    
                    # 构建路径组件列表
                    curr_path_list = rel_path.split(os.sep) if rel_path else []
                    
                    # 优化树结构构建
                    curr_dict = case_yml_info_dict
                    for k in curr_path_list[:-1]:  # 处理中间路径组件
                        curr_dict = curr_dict.setdefault(k, {})
                    
                    # 处理最后一级路径
                    last_key = curr_path_list[-1] if curr_path_list else ""
                    value = config.get(label)
                    
                    # 列表类型特殊处理
                    if isinstance(value, list):
                        # 创建路径顺序映射 {元素: 索引}
                        path_order = {}
                        for idx, item in enumerate(curr_path_list):
                            if item not in path_order:
                                path_order[item] = idx
                        
                        # 使用自定义排序规则进行排序
                        value = sorted(value, key=lambda x: self._custom_sort(x, path_order))
                    
                    # 更新到字典
                    if last_key:
                        curr_dict[last_key] = value
                    else:
                        # 根目录下的case.yml
                        case_yml_info_dict.update(value if isinstance(value, dict) else {label: value})
        
        # 打印树前检查是否找到匹配用例
        if self.filter_case_name and not found_target:
            print(f"\n\t{RED}未找到用例 '{self.filter_case_name}'{RESET}\n")
            return
        
        # 打印最终的树状结构
        self.print_tree(case_yml_info_dict, label=label)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process case.yml files to add or remove label value.")
    parser.add_argument("-p", "--path", type=str, required=True, help="The path to search for case.yml files.")
    parser.add_argument("-l", "--label", type=str, required=True, help="Label to change.")
    parser.add_argument("-a", "--add", nargs='*', 
                        help="Values to add, split by ' '. Notice: '123' is diff from 123, each element for dict type should be formatted as k:v.")
    parser.add_argument("-r", "--remove", nargs='*',
                        help="Values to remove, split by ' '. Notice: '123' is diff from 123, each element for dict type should be formatted as k:v.")
    parser.add_argument("-s", "--show",
                        nargs='?', const="",
                        help="Show label value: without arg shows all, with case_name shows specific case"
                        )
    args = parser.parse_args()
    
    if args.label:
        if args.label in LABEL_TYPE_MAP.keys():
            if args.show is None and not args.add and not args.remove:
                print(f"{YELLOW}Warning: No action has been selected.{RESET}")
            if args.add or args.remove:
                UpdateOperation(args.path, args.label, args.add, args.remove)
            if args.show is not None:
                ShowOperation(args.path, args.label, filter_case_name=args.show if args.show != "" else None)
        else:
            raise ValueError(f"{RED}Error label - {{{args.label}}} not in {LABEL_TYPE_MAP.keys()}{RESET}")
    else:
        raise ValueError(f"{RED}Error: Select one label in {LABEL_TYPE_MAP.keys()}{RESET}")