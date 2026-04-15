import os
import json
import time
from collections import defaultdict
from run_log import log

class TaskScheduler:
    """
    测试任务调度器，管理执行时间的记录和优化任务调度
    
    功能:
    1. 记录测试用例执行时间历史
    2. 提供预估执行时间
    3. 实现LPT(最长处理时间优先)调度算法
    """
    
    def __init__(self, data_file):
        """
        初始化调度器
        
        参数:
            data_file: 历史时间数据的存储文件, 格式为json文件
        """
        self.data_file = data_file
        self.historical_times = defaultdict(list)
        self._load_data()

    def __del__(self):
        """析构时保存数据"""
        self.save_data()
        
    def _load_data(self):
        """从文件加载历史执行时间数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    loaded_data = json.load(f)
                    # 转换为defaultdict(list)
                    for case_name, times in loaded_data.items():
                        self.historical_times[case_name] = times
            except Exception as e:
                log(f"加载历史时间数据失败: {str(e)}")
    
    def save_data(self):
        """保存历史执行时间数据到文件"""
        try:
            with open(self.data_file, 'w') as f:
                # 使用标准dict保存数据
                json.dump(dict(self.historical_times), f, indent=2)
        except Exception as e:
            log(f"保存历史时间数据失败: {str(e)}")
    
    def get_estimated_time(self, case_name: str):
        """
        获取用例的预估执行时间
        
        参数:
            case_name: 测试用例名称
            
        返回:
            float: 预估执行时间（秒）
        """
        times = self.historical_times.get(case_name, [])
        
        if not times:
            return 0  # 无历史数据
        
        # 最近3次的中位数
        recent_times = times[-3:] if len(times) >= 3 else times
        
        # 如果只有一次记录，直接返回
        if len(recent_times) == 1:
            return recent_times[0]
        
        # 排序获取中位数
        sorted_times = sorted(recent_times)
        mid = len(sorted_times) // 2
        if len(sorted_times) % 2 == 0:
            return (sorted_times[mid-1] + sorted_times[mid]) / 2
        else:
            return sorted_times[mid]
    
    def update_execution_time(self, case_name: str, duration: float):
        """
        更新用例执行时间（记录本次执行结果）
        
        参数:
            case_name: 测试用例名称
            duration: 本次执行耗时（秒）
        """
        # 维护最近10次执行记录
        records = self.historical_times.get(case_name, [])
        records.append(duration)
        if len(records) > 10:
            records.pop(0)
        
        self.historical_times[case_name] = records
    
    def schedule_LPT(self, group_configs: list[dict]) -> list[dict]:
        """
        使用LPT算法优化任务调度顺序
        
        参数:
            group_configs: 待调度的测试用例配置列表
            
        返回:
            list: 优化排序后的测试用例配置列表
        """
        # 为每个配置添加预估时间属性
        for config in group_configs:
            config['_estimated_time'] = self.get_estimated_time(config['case_name'])
        
        # 按预估时间降序排序 (最长任务优先)
        return sorted(
            group_configs, 
            key=lambda x: x.get('_estimated_time', 0), 
            reverse=True
        )
    
    def record_case_execution(self, config: dict, result: dict):
        """
        记录用例执行时间（在用例执行完成后调用）
        
        参数:
            config: 测试用例配置
            result: 测试结果（应包含执行时间）
        """
        case_name = config.get('case_name')
        if not case_name:
            return
        
        # 从结果中获取执行时间
        execution_time = result.get('execution_time', None)
        if execution_time is not None:
            self.update_execution_time(case_name, execution_time)