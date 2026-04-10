#!/usr/bin/env python3
"""
目录监控与清理脚本 - 增强版
功能：监控指定目录下的第一层子目录数量，超过阈值时删除指定天数前的子目录

用法：python directory_cleaner.py --path <路径> --max-count <数量> --days <天数>
示例：python directory_cleaner.py -p /var/log/backups -c 20 -d 30 -dr
"""

import os
import sys
import argparse
import time
import logging
import heapq
import shutil
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('directory_cleaner')

def get_subdirectories(dir_path: str) -> list:
    """
    获取指定路径下所有第一层子目录（非递归）
    返回格式：[(目录路径, 最后修改时间)]
    """
    subdirs = []
    try:
        for entry in os.scandir(dir_path):
            if entry.is_dir(follow_symlinks=False):
                # 获取目录最后修改时间
                mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                subdirs.append((entry.path, mtime))
    except OSError as e:
        logger.error(f"无法扫描目录 {dir_path}: {str(e)}")
    
    return subdirs

def clean_old_subdirectories(dir_path: str, max_dirs: int, days_threshold: int) -> int:
    """
    清理旧目录的主函数
    返回删除的目录数量
    """
    subdirs = get_subdirectories(dir_path)
    if not subdirs:
        logger.info(f"目录 {dir_path} 中没有子目录")
        return 0

    threshold_time = datetime.now() - timedelta(days=days_threshold)
    current_count = len(subdirs)
    
    if current_count <= max_dirs:
        logger.info(f"无需清理: 当前 {current_count} 个目录 ≤ 阈值 {max_dirs}")
        return 0

    # === 精确计算需删除数量 ===
    need_delete = current_count - max_dirs
    logger.info(f"需清理数量: {need_delete} (当前 {current_count} - 阈值 {max_dirs})")

    # === 使用堆获取最旧的N个目录 ===
    old_dirs = [
        (mtime, path) for path, mtime in subdirs 
        if mtime < threshold_time
    ]
    
    # 按时间升序获取最旧的need_delete个目录（堆优化）
    oldest_dirs = heapq.nsmallest(
        need_delete,
        old_dirs,
        key=lambda x: x[0]
    )
    
    if not oldest_dirs:
        logger.info("没有符合条件的旧目录可删除（均不超过天数阈值）")
        return 0

    # 转换为 (路径, 修改时间) 格式并排序（旧→新）
    sorted_dirs = sorted(
        [(path, mtime) for mtime, path in oldest_dirs],
        key=lambda x: x[1]
    )

    deleted_count = 0
    for path, mtime in sorted_dirs:
        try:
            shutil.rmtree(path)
            logger.info(f"已删除: {path} (修改时间: {mtime.strftime('%Y-%m-%d')})")
            deleted_count += 1
        except Exception as e:
            logger.error(f"删除失败 {path}: {str(e)}")

    remaining = current_count - deleted_count
    logger.info(f"清理完成: 删除 {deleted_count} 个目录，剩余 {remaining} 个")
    return deleted_count

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='目录清理工具：当子目录超过数量阈值时，删除超过指定天数的旧目录',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加参数选项
    parser.add_argument('--path', '-p', 
                        required=True,
                        help='要监控的目标目录路径')
    parser.add_argument('--max-count', '-c', 
                        type=int, 
                        required=True,
                        help='允许的最大子目录数量')
    parser.add_argument('--days', '-d', 
                        type=int, 
                        default=30,
                        help='天数阈值，超过此天数的目录会被视为旧目录')
    parser.add_argument('--dry-run', '-dr',
                        action='store_true', 
                        help='模拟运行模式，仅显示将要进行的操作而不实际删除')
    
    # 解析参数
    args = parser.parse_args()
    
    # 验证参数有效性
    if not os.path.isdir(args.path):
        logger.error(f"错误: 目录不存在 - {args.path}")
        sys.exit(1)
        
    if args.max_count <= 0:
        logger.error("错误: max-count 必须为正整数")
        sys.exit(1)
        
    if args.days <= 0:
        logger.error("错误: days 必须为正整数")
        sys.exit(1)
    
    # 显示当前配置
    config_info = (
        f"清理配置:\n"
        f"  目标目录: {args.path}\n"
        f"  最大数量: {args.max_count}\n"
        f"  天数阈值: {args.days}\n"
        f"  模式: {'模拟运行' if args.dry_run else '实际执行'}"
    )
    logger.info(config_info)
    
    # 处理模拟运行模式
    if args.dry_run:
        logger.info("==== 模拟运行模式 ==== (不会实际删除任何目录)")
        subdirs = get_subdirectories(args.path)
        time_threshold = datetime.now() - timedelta(days=args.days)
        
        if len(subdirs) > args.max_count:
            sorted_dirs = sorted(subdirs, key=lambda x: x[1])
            candidates = []
            
            for path, mtime in sorted_dirs:
                if mtime < time_threshold:
                    candidates.append(f"{path} (修改时间: {mtime.strftime('%Y-%m-%d')})")
            
            # 计算需要删除的数量
            to_delete = min(len(candidates), len(subdirs) - args.max_count)
            
            if to_delete > 0:
                logger.info(f"将被删除的目录 ({to_delete} 个):")
                for dir_info in candidates[:to_delete]:
                    logger.info(f"  - {dir_info}")
            else:
                logger.info("没有符合条件的旧目录可删除")
        else:
            logger.info(f"无需清理: 当前 {len(subdirs)} 个目录 不超过阈值 {args.max_count}")
    else:
        # 实际执行清理
        try:
            deleted = clean_old_subdirectories(
                dir_path=args.path,
                max_dirs=args.max_count,
                days_threshold=args.days
            )
        except Exception as e:
            logger.exception(f"执行过程中发生意外错误: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 6):
        print("错误: 需要Python 3.6或更高版本")
        sys.exit(1)
    
    main()