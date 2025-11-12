import unittest

from ais_bench.benchmark.datasets.ifeval.instructions_registry import (
    conflict_make,
    INSTRUCTION_DICT,
    INSTRUCTION_CONFLICTS,
)


class TestInstructionsRegistry(unittest.TestCase):
    def test_conflict_make(self):
        """测试 conflict_make 函数确保冲突是对称的"""
        test_conflicts = {
            'A': {'B'},
            'B': set(),
            'C': {'D'},
            'D': set(),  # 添加 D 的初始条目
        }
        result = conflict_make(test_conflicts)
        
        # 验证每个指令都与自己冲突
        self.assertIn('A', result['A'])
        self.assertIn('B', result['B'])
        self.assertIn('C', result['C'])
        self.assertIn('D', result['D'])
        
        # 验证如果 A 与 B 冲突，则 B 也与 A 冲突
        self.assertIn('A', result['B'])
        self.assertIn('B', result['A'])
        
        # 验证如果 C 与 D 冲突，则 D 也与 C 冲突
        self.assertIn('C', result['D'])
        self.assertIn('D', result['C'])

    def test_instruction_dict_not_empty(self):
        """验证 INSTRUCTION_DICT 不为空"""
        self.assertGreater(len(INSTRUCTION_DICT), 0)

    def test_instruction_conflicts_not_empty(self):
        """验证 INSTRUCTION_CONFLICTS 不为空"""
        self.assertGreater(len(INSTRUCTION_CONFLICTS), 0)


if __name__ == "__main__":
    unittest.main()

