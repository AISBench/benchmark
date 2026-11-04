import json
import random
import os


class DataPicker:
    def __init__(self, jsonl_file, record_file="picked_ids.txt", prefix_flag=0):
        """
        初始化挑选器

        Args:
            jsonl_file: GSM8K数据集文件路径
            record_file: 记录已挑选数据编号的文件路径
        """
        self.jsonl_file = jsonl_file
        self.record_file = record_file
        self.total_lines = self._count_lines()
        self.picked_ids = self._load_picked_ids()
        self.prefix_flag = prefix_flag

    def _count_lines(self):
        """统计jsonl文件总行数"""
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _load_picked_ids(self):
        """加载已挑选的编号记录"""
        if os.path.exists(self.record_file):
            with open(self.record_file, 'r', encoding='utf-8') as f:
                return set(int(line.strip()) for line in f if line.strip())
        return set()

    def _save_picked_ids(self):
        """保存已挑选的编号记录"""
        with open(self.record_file, 'w', encoding='utf-8') as f:
            for pid in sorted(self.picked_ids):
                f.write(f"{pid}\n")

    def get_unpicked_ids(self):
        """获取所有未被挑选过的编号"""
        all_ids = set(range(self.total_lines))
        return list(all_ids - self.picked_ids)

    def pick_one(self):
        """
        随机挑选一条数据

        Returns:
            挑选的数据内容，如果没有未挑选的数据则返回None
        """

        # 获取可选的编号列表
        if self.prefix_flag == 1:
            # 不重复模式：从未挑选的编号中选择
            available_ids = self.get_unpicked_ids()
            if not available_ids:
                return None
            # 随机选择一个未挑选的编号
            selected_id = random.choice(available_ids)
            # 记录已挑选的编号
            self.picked_ids.add(selected_id)
            self._save_picked_ids()
        else:
            # 可重复模式：从所有编号中随机选择，不记录编号
            selected_id = random.randint(0, self.total_lines - 1)

        # 读取对应编号的数据
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == selected_id:
                    data = json.loads(line)
                    break

        return data['question']

    # def reset(self):
    #     """重置记录（清空已挑选记录）"""
    #     self.picked_ids.clear()
    #     self._save_picked_ids()
