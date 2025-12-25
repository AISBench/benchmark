import os
import sys
from threading import Lock

class Logger:
    """
    单例日志记录器类
    
    特性:
    - 全局唯一实例
    - 线程安全的写入操作
    - 自动创建日志文件
    - 支持任意位置调用写入
    - 简化使用接口
    """
    
    # 单例实例
    _instance = None
    # 创建实例的锁
    _lock = Lock()
    # 写入锁
    _write_lock = Lock()
    
    def __init__(self, log_dir: str):
        """
        创建日志实例(应通过get_instance调用)
        
        参数:
            log_dir: 日志目录路径
        """
        if not hasattr(self, 'log_dir'):  # 防止重复初始化
            self.log_dir = log_dir
            
            # 验证并创建日志目录
            os.makedirs(log_dir, exist_ok=True)
            if not os.path.isdir(log_dir):
                raise ValueError(f"无效的日志目录: {log_dir}")
                
            # 日志文件路径
            self.log_path = os.path.join(log_dir, "run.log")
            
            # 打开日志文件
            self._file = open(self.log_path, "a", encoding="utf-8")
            
            # 写入初始分隔符
            self._write_raw("\n" + "=" * 60 + "\n")
            self._write_raw("日志系统初始化完成\n")
            self._write_raw("=" * 60 + "\n\n")
    
    @classmethod
    def get_instance(cls, log_dir: str = None):
        """
        获取日志实例(单例)
        
        参数:
            log_dir: 首次调用时提供日志目录路径
                    后续调用可忽略
        
        返回:
            Logger实例
            
        异常:
            首次调用未提供log_dir抛出ValueError
        """
        if cls._instance is None:
            with cls._lock:
                # 双重检查确保单例
                if cls._instance is None:
                    if log_dir is None:
                        raise ValueError("首次使用必须提供log_dir参数")
                    cls._instance = cls(log_dir)
        
        return cls._instance
    
    @classmethod
    def is_initialized(cls):
        """
        检查日志实例是否已初始化
        
        返回:
            bool: True表示已初始化
        """
        return cls._instance is not None
    
    def write(self, *args, sep: str = ' ', end: str = '\n'):
        """
        写入日志消息(支持多个参数)
        
        参数:
            *args: 要记录的任意数量参数
            sep: 参数间的分隔符(默认为空格)
            end: 日志行结束符(默认为换行符)
        """
        # 确保实例已初始化
        if not self.is_initialized():
            raise RuntimeError("日志实例未初始化")
            
        with self._write_lock:
            # 格式化消息
            message = sep.join(str(arg) for arg in args) + end
            self._write_raw(message)
    
    def _write_raw(self, content: str):
        """底层写操作"""
        self._file.write(content)
        self._file.flush()  # 确保写入磁盘
    
    def close(self):
        """关闭日志文件(通常不需要手动调用)"""
        if hasattr(self, '_file') and self._file:
            self._write_raw("\n" + "=" * 60 + "\n")
            self._write_raw("日志会话结束\n")
            self._file.close()
            self._file = None
    
    def __del__(self):
        """析构时关闭文件"""
        self.close()

# 全局快捷函数
def log(*args, sep: str = ' ', end: str = '\n', flush: bool = False, print_enable: bool = True):
    """
    全局日志函数(支持多个参数)
    
    参数:
        *args: 要记录的任意数量参数
        sep: 参数间的分隔符(默认为空格)
        end: 日志行结束符(默认为换行符)
    """
    try:
        if Logger.is_initialized():
            Logger.get_instance().write(*args, sep=sep, end=end)
        if print_enable:
            print(*args, sep=sep, end=end, flush=flush)
    except Exception as e:
        # 最后的回退方案
        print("[LOG ERROR]", *args, sep=sep, end=end, file=sys.stderr)
        print(f"Error Detail: {e}")

def initialize_logging(log_dir:str):
    """应用启动时调用此函数初始化日志"""
    try:
        # 选择日志目录
        Logger.get_instance(log_dir)
    except Exception as e:
        print(f"日志初始化失败: {e}", file=sys.stderr)
        sys.exit(1)