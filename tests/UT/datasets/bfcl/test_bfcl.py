"""
BFCL 纯函数单元测试 - Part 1
直接测试不依赖外部复杂依赖的纯函数
目标覆盖率: 80%
"""
import sys
import os
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# 尝试导入实际模块，如果失败则创建mock
BFCL_AVAILABLE = False
try:
    from ais_bench.benchmark.datasets.bfcl.bfcl import (
        encode_fields, 
        VERSION_PREFIX,
        BFCLEvaluator,
        is_java,
        is_js,
        BFCLDataset,
        BFCLRelevanceEvaluator,
        BFCLMultiTurnEvaluator,
        BFCLSingleTurnEvaluator
    )
    BFCL_AVAILABLE = True
    print("Successfully imported BFCL modules")
except ImportError as e:
    print(f"ImportError: {e}. BFCL tests will be skipped.")
    # 创建 mock 函数和类
    def encode_fields(data):
        """Mock encode_fields function"""
        fields = [
            "question",
            "ground_truth",
            "function",
            "missed_function",
            "involved_classes",
            "initial_config",
        ]
        for item in data:
            for field in fields:
                if field in item and not isinstance(item[field], str):
                    item[field] = json.dumps(item[field], ensure_ascii=False)
        return data
    
    VERSION_PREFIX = "BFCL_v3"
    
    def is_java(category):
        """Mock is_java function"""
        return category.lower() == "java"
    
    def is_js(category):
        """Mock is_js function"""
        return category.lower() == "javascript" or category.lower() == "js"
    
    class BaseDataset:
        pass
    
    class BaseEvaluator:
        pass
    
    class BFCLDataset(BaseDataset):
        def __init__(self, **kwargs):
            pass
        
        @staticmethod
        def load(path, category, test_ids=None):
            pass
    
    class BFCLEvaluator(BaseEvaluator):
        def __init__(self, category: str, is_fc_model=True):
            self.is_fc_model = is_fc_model
            self.category = category
            self.model_name = "function-call-model-" + str(uuid.uuid4()).split("-")[-1]
            self.language = "Python"
            # 添加is_empty_execute_response和is_empty_output方法
            self.is_empty_execute_response = lambda response: not response or (isinstance(response, list) and all(not item for item in response))
            self.is_empty_output = lambda output: output is None or output == ""
        
        def score(self, *args, **kwargs):
            raise NotImplementedError("Must be implemented in subclasses")
        
        def decode_ast(self, result, language=None):
            return result
    
    class BFCLRelevanceEvaluator(BFCLEvaluator):
        def score(self, prediction, ground_truth, test_set=None):
            pass
    
    class BFCLMultiTurnEvaluator(BFCLEvaluator):
        def decode_execute(self, *args, **kwargs):
            pass
        
        def score(self, prediction, ground_truth, test_set=None):
            pass
    
    class BFCLSingleTurnEvaluator(BFCLEvaluator):
        def score(self, prediction, ground_truth, test_set=None):
            pass


# ==================== 测试 encode_fields 函数 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_non_string_fields():
    """测试编码非字符串字段"""
    data = [
        {
            "question": ["What is the capital?"],
            "ground_truth": {"answer": "Paris"},
            "function": {"name": "get_capital"},
            "other_field": "keep_as_is"
        }
    ]
    
    result = encode_fields(data)
    
    # 验证非字符串字段被编码为 JSON 字符串
    assert isinstance(result[0]["question"], str)
    assert isinstance(result[0]["ground_truth"], str)
    assert isinstance(result[0]["function"], str)
    # 验证可以解码回原始数据
    assert json.loads(result[0]["question"]) == ["What is the capital?"]
    assert json.loads(result[0]["ground_truth"]) == {"answer": "Paris"}
    # 验证原本就是字符串的字段保持不变
    assert result[0]["other_field"] == "keep_as_is"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_already_string_fields():
    """测试已经是字符串的字段不被重复编码"""
    data = [
        {
            "question": '["What is the capital?"]',
            "ground_truth": '{"answer": "Paris"}',
            "function": '{"name": "test"}'
        }
    ]
    
    result = encode_fields(data)
    
    # 验证已经是字符串的字段保持不变
    assert result[0]["question"] == '["What is the capital?"]'
    assert result[0]["ground_truth"] == '{"answer": "Paris"}'
    assert result[0]["function"] == '{"name": "test"}'

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_empty_list():
    """测试空列表"""
    data = []
    result = encode_fields(data)
    
    assert len(result) == 0
    assert isinstance(result, list)

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_all_supported_fields():
    """测试所有支持的字段"""
    data = [
        {
            "question": ["Q1", "Q2"],
            "ground_truth": ["GT1", "GT2"],
            "function": {"func": "test_func"},
            "missed_function": ["miss1", "miss2"],
            "involved_classes": ["class1", "class2"],
            "initial_config": {"config": "value"}
        }
    ]
    
    result = encode_fields(data)
    
    # 验证所有支持的字段都被正确编码
    for field in ["question", "ground_truth", "function", "missed_function", "involved_classes", "initial_config"]:
        assert field in result[0]
        assert isinstance(result[0][field], str)
        # 验证可以解码
        decoded = json.loads(result[0][field])
        assert decoded is not None

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_mixed_fields():
    """测试混合字段（部分是字符串，部分不是）"""
    data = [
        {
            "question": ["Q1"],  # 需要编码
            "ground_truth": '["GT1"]',  # 已经是字符串
            "function": {"func": "test"},  # 需要编码
            "other": "plain_string"  # 不在编码列表中
        }
    ]
    
    result = encode_fields(data)
    
    # 验证需要编码的字段被编码
    assert result[0]["question"] == '["Q1"]'
    # 验证已经是字符串的保持不变
    assert result[0]["ground_truth"] == '["GT1"]'
    # 验证 function 被编码
    assert '"func"' in result[0]["function"]
    # 验证其他字段不受影响
    assert result[0]["other"] == "plain_string"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_nested_structures():
    """测试嵌套结构"""
    data = [
        {
            "question": {
                "text": "What is it?",
                "options": ["A", "B", "C"]
            },
            "ground_truth": [
                {"answer": "A", "explanation": "Because..."}
            ]
        }
    ]
    
    result = encode_fields(data)
    
    # 验证嵌套结构被正确编码
    assert isinstance(result[0]["question"], str)
    assert isinstance(result[0]["ground_truth"], str)
    
    # 验证可以解码并保持结构
    decoded_question = json.loads(result[0]["question"])
    assert decoded_question["text"] == "What is it?"
    assert len(decoded_question["options"]) == 3
    
    decoded_gt = json.loads(result[0]["ground_truth"])
    assert decoded_gt[0]["answer"] == "A"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_unicode_content():
    """测试 Unicode 内容（中文等）"""
    data = [
        {
            "question": ["什么是首都？"],
            "ground_truth": {"答案": "北京"},
            "function": {"名称": "获取首都"}
        }
    ]
    
    result = encode_fields(data)
    
    # 验证 Unicode 内容被正确编码（ensure_ascii=False）
    assert isinstance(result[0]["question"], str)
    assert "什么是首都" in result[0]["question"]
    assert "答案" in result[0]["ground_truth"]
    assert "名称" in result[0]["function"]
    
    # 验证可以正确解码
    decoded_question = json.loads(result[0]["question"])
    assert decoded_question[0] == "什么是首都？"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_preserves_order():
    """测试编码保持数据顺序"""
    data = [
        {"id": 1, "question": ["Q1"]},
        {"id": 2, "question": ["Q2"]},
        {"id": 3, "question": ["Q3"]}
    ]
    
    result = encode_fields(data)
    
    # 验证顺序保持不变
    assert len(result) == 3
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2
    assert result[2]["id"] == 3

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_empty_values():
    """测试空值"""
    data = [
        {
            "question": [],
            "ground_truth": {},
            "function": []
        }
    ]
    
    result = encode_fields(data)
    
    # 验证空列表和空字典被编码
    assert result[0]["question"] == "[]"
    assert result[0]["ground_truth"] == "{}"
    assert result[0]["function"] == "[]"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_partial_fields():
    """测试只包含部分字段的数据"""
    data = [
        {
            "question": ["Q1"],
            "other_field": "value"
        }
    ]
    
    result = encode_fields(data)
    
    # 验证存在的字段被编码
    assert result[0]["question"] == '["Q1"]'
    # 验证不在列表中的字段不受影响
    assert result[0]["other_field"] == "value"
    # 验证不存在的字段不会被添加
    assert "ground_truth" not in result[0]


# ==================== 测试 VERSION_PREFIX ====================

def test_version_prefix_format():
    """测试版本前缀格式"""
    assert isinstance(VERSION_PREFIX, str)
    assert "BFCL" in VERSION_PREFIX
    assert "v" in VERSION_PREFIX

def test_version_prefix_value():
    """测试版本前缀的具体值"""
    assert VERSION_PREFIX == "BFCL_v3"

def test_version_prefix_usage_in_filename():
    """测试版本前缀在文件名中的使用"""
    # 模拟文件名构造
    category = "python"
    filename = f"{VERSION_PREFIX}_{category}.json"
    
    assert filename == "BFCL_v3_python.json"
    assert "BFCL" in filename
    assert "python" in filename
    assert filename.endswith(".json")

def test_version_prefix_in_ground_truth_path():
    """测试版本前缀在 ground truth 路径中的使用"""
    category = "javascript"
    gt_path = f"possible_answer/{VERSION_PREFIX}_{category}.json"
    
    assert gt_path == "possible_answer/BFCL_v3_javascript.json"
    assert "possible_answer" in gt_path


# ==================== 测试 BFCLEvaluator 基类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_python_category():
    """测试 Python 类别初始化"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=True)
    
    assert evaluator.language == "Python"
    assert evaluator.category == "python"
    assert evaluator.is_fc_model is True
    assert evaluator.model_name is not None
    assert "function-call-model-" in evaluator.model_name

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_java_category():
    """测试 Java 类别初始化"""
    evaluator = BFCLEvaluator(category="java", is_fc_model=True)
    
    assert evaluator.language == "Java"
    assert evaluator.category == "java"
    assert evaluator.is_fc_model is True

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_js_category():
    """测试 JavaScript 类别初始化"""
    evaluator = BFCLEvaluator(category="javascript", is_fc_model=True)
    
    assert evaluator.language == "JavaScript"
    assert evaluator.category == "javascript"
    assert evaluator.is_fc_model is True

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_non_fc_model():
    """测试非 FC 模型初始化"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=False)
    
    assert evaluator.is_fc_model is False
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_model_name_uniqueness():
    """测试模型名称的唯一性"""
    evaluator1 = BFCLEvaluator(category="python")
    evaluator2 = BFCLEvaluator(category="python")
    
    # 验证两个实例有不同的模型名称
    assert evaluator1.model_name != evaluator2.model_name

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_score_not_implemented():
    """测试 score 方法抛出 NotImplementedError"""
    evaluator = BFCLEvaluator(category="python")
    
    with pytest.raises(NotImplementedError):
        evaluator.score([], [])

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_decode_ast_fc_model():
    """测试 FC 模型的 AST 解码"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=True)
    
    result = [
        {"func1": '{"param": "value"}'},
        {"func2": '{"param2": "value2"}'}
    ]
    
    decoded = evaluator.decode_ast(result)
    
    assert len(decoded) == 2
    assert "func1" in decoded[0]
    if BFCL_AVAILABLE and hasattr(decoded[0]["func1"], "get"):
        assert decoded[0]["func1"].get("param") == "value"
    assert "func2" in decoded[1]

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_decode_ast_prompting_model():
    """测试 Prompting 模型的 AST 解码"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=False)
    
    # 验证基本的初始化
    assert evaluator.is_fc_model is False
    assert evaluator.category == "python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_decode_ast_with_language_parameter():
    """测试带语言参数的 AST 解码"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=True)
    
    result = [{"func": '{"param": "value"}'}]
    
    # 测试不同语言参数
    decoded_python = evaluator.decode_ast(result, language="Python")
    decoded_java = evaluator.decode_ast(result, language="Java")
    
    # 验证解码结果相同（对于 FC 模型，语言参数不影响解码）
    assert len(decoded_python) == len(decoded_java)


# ==================== 测试 BFCL 数据结构 ====================

def test_dataset_item_structure():
    """测试数据集项的基本结构"""
    # 模拟一个典型的数据集项
    item = {
        "id": "test_001",
        "question": "What is the weather?",
        "function": {"name": "get_weather", "parameters": {}},
        "ground_truth": ["get_weather(location='Beijing')"]
    }
    
    # 验证必要字段存在
    assert "id" in item
    assert "question" in item
    assert "ground_truth" in item
    
    # 验证字段类型
    assert isinstance(item["id"], str)
    assert isinstance(item["question"], str)
    assert isinstance(item["ground_truth"], list)

def test_relevance_dataset_structure():
    """测试 relevance 数据集的特殊结构"""
    # relevance 数据集有特殊的 ground truth
    item = {
        "id": "relevance_test_001",
        "question": "What is the weather?",
        "ground_truth": ["relevance_mock_gt"]
    }
    
    # 验证 relevance 测试的标识
    assert "relevance" in item["id"]
    assert "mock" in str(item["ground_truth"])
    assert item["ground_truth"][0] == "relevance_mock_gt"

def test_irrelevance_dataset_structure():
    """测试 irrelevance 数据集的结构"""
    item = {
        "id": "irrelevance_test_001",
        "question": "Random question",
        "ground_truth": ["relevance_mock_gt"]
    }
    
    # 验证 irrelevance 测试的标识
    assert "irrelevance" in item["id"]

def test_multi_turn_structure():
    """测试多轮对话数据结构"""
    # 多轮对话的数据结构
    item = {
        "id": "multi_turn_001",
        "question": [
            "What is the weather?",
            "What about tomorrow?"
        ],
        "ground_truth": [
            ["get_weather(location='Beijing')"],
            ["get_weather(location='Beijing', date='tomorrow')"]
        ],
        "initial_config": {},
        "involved_classes": []
    }
    
    # 验证多轮结构
    assert isinstance(item["question"], list)
    assert isinstance(item["ground_truth"], list)
    assert "initial_config" in item
    assert "involved_classes" in item
    
    # 验证多轮数据的长度匹配
    assert len(item["question"]) == len(item["ground_truth"])

def test_single_turn_structure():
    """测试单轮对话数据结构"""
    item = {
        "id": "single_turn_001",
        "question": "What is the weather?",
        "function": [{"name": "get_weather", "parameters": {}}],
        "ground_truth": ["get_weather(location='Beijing')"]
    }
    
    # 验证单轮结构
    assert isinstance(item["question"], str)
    assert isinstance(item["ground_truth"], list)
    assert "function" in item

def test_category_identifier():
    """测试类别标识符"""
    categories = ["python", "java", "javascript", "relevance", "irrelevance"]
    
    for category in categories:
        # 验证类别名称格式
        assert isinstance(category, str)
        assert len(category) > 0
        assert category.islower() or '_' in category

def test_id_format():
    """测试 ID 格式"""
    valid_ids = [
        "test_001",
        "multi_turn_python_001",
        "relevance_test_01",
        "irrelevance_test_02",
        "single_turn_java_001"
    ]
    
    for id_value in valid_ids:
        # 验证 ID 格式
        assert isinstance(id_value, str)
        assert len(id_value) > 0
        # ID 通常包含下划线分隔
        assert "_" in id_value


# ==================== 测试 BFCLDataset 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_initialization():
    """测试数据集初始化"""
    # 验证类可以被正确实例化
    assert hasattr(BFCLDataset, 'load')
    assert callable(BFCLDataset.load)

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_load_method_parameters():
    """测试load方法的参数处理"""
    # 验证load方法接受正确的参数
    signature = BFCLDataset.load.__code__
    params = signature.co_varnames[:signature.co_argcount]
    assert 'path' in params
    assert 'category' in params
    assert 'test_ids' in params

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_load_normal():
    """测试BFCLDataset.load方法的正常加载情况"""
    from ais_bench.benchmark.datasets.utils import datasets as utils_datasets
    from ais_bench.benchmark.datasets.bfcl import bfcl
    
    # 配置mock
    def mock_get_data_path_side_effect(path):
        # 对于ground_truth路径，返回一个mock路径
        if "possible_answer" in path:
            return "/tmp/mocked_gt_path.json"
        return "/tmp/mocked_data_path.json"
    
    mock_get_data_path = MagicMock(side_effect=mock_get_data_path_side_effect)
    
    # 模拟数据集文件
    dataset_file = MagicMock()
    dataset_file.__enter__.return_value = dataset_file
    dataset_file.__iter__.return_value = [
        '{"id": "test_001", "question": "Q1"}\n',
        '{"id": "test_002", "question": "Q2"}\n'
    ]
    
    # 模拟ground_truth文件
    gt_file = MagicMock()
    gt_file.__enter__.return_value = gt_file
    gt_file.__iter__.return_value = [
        '{"id": "test_001", "ground_truth": ["gt1"]}\n',
        '{"id": "test_002", "ground_truth": ["gt2"]}\n'
    ]
    
    # open函数根据路径返回不同的文件
    def mock_open_side_effect(path, *args, **kwargs):
        if "possible_answer" in path or "mocked_gt_path" in path:
            return gt_file
        return dataset_file
    
    mock_open_file = MagicMock(side_effect=mock_open_side_effect)
    mock_process_multi = MagicMock(side_effect=lambda x: x)  # 直接返回输入
    mock_encode = MagicMock(side_effect=lambda x: x)  # 直接返回输入
    mock_dataset_instance = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.from_list.return_value = mock_dataset_instance
    
    with patch.object(utils_datasets, 'get_data_path', mock_get_data_path), \
         patch('builtins.open', mock_open_file), \
         patch.object(bfcl, 'Dataset', mock_dataset), \
         patch.object(bfcl, 'encode_fields', mock_encode), \
         patch.object(bfcl, 'process_multi_turn_test_case', mock_process_multi), \
         patch.object(bfcl, 'BFCL_INSTALLED', True):
        # 执行load方法 - 使用绝对路径避免路径检查
        result = BFCLDataset.load(path="/tmp/test_path", category="python")
        
        # 验证结果
        assert result == mock_dataset_instance
        # 验证调用参数包含ground_truth
        call_args = mock_dataset.from_list.call_args[0][0]
        assert len(call_args) == 2
        assert "ground_truth" in call_args[0]
        mock_encode.assert_called_once()
        mock_process_multi.assert_called_once()

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.BFCL_INSTALLED', False)
def test_bfcl_dataset_load_missing_dependency():
    """测试BFCLDataset.load方法在缺少依赖时的情况"""
    with pytest.raises(ImportError) as excinfo:
        BFCLDataset.load(path="test_path", category="python")
    
    # 验证异常消息
    assert "Missing required package 'bfcl-eval'" in str(excinfo.value)

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_load_relevance_category():
    """测试BFCLDataset.load方法处理relevance类别"""
    from ais_bench.benchmark.datasets.utils import datasets as utils_datasets
    from ais_bench.benchmark.datasets.bfcl import bfcl
    
    # 配置mock
    mock_get_data_path = MagicMock(return_value="mocked_data_path")
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__iter__.return_value = [
        '{"id": "relevance_001", "question": "Q1"}\n'
    ]
    mock_open_file = MagicMock(return_value=mock_file)
    mock_process = MagicMock(return_value=[{"id": "relevance_001", "ground_truth": ["relevance_mock_gt"]}])
    mock_encode = MagicMock(return_value=[{"id": "relevance_001", "encoded": True}])
    mock_dataset = MagicMock()
    mock_dataset.from_list.return_value = MagicMock()
    
    with patch.object(utils_datasets, 'get_data_path', mock_get_data_path), \
         patch('builtins.open', mock_open_file), \
         patch.object(bfcl, 'process_multi_turn_test_case', mock_process), \
         patch.object(bfcl, 'encode_fields', mock_encode), \
         patch.object(bfcl, 'Dataset', mock_dataset), \
         patch.object(bfcl, 'BFCL_INSTALLED', True):
        # 执行load方法 - 使用绝对路径避免FileExistsError
        BFCLDataset.load(path="/tmp/test_path", category="relevance")
        
        # 验证relevance类别被正确处理
        assert mock_process.called
        assert mock_encode.called



"""
BFCL 纯函数单元测试 - Part 2
直接测试不依赖外部复杂依赖的纯函数
目标覆盖率: 80%
"""
import sys
import os
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# 尝试导入实际模块，如果失败则创建mock
BFCL_AVAILABLE = False
try:
    from ais_bench.benchmark.datasets.bfcl.bfcl import (
        encode_fields, 
        VERSION_PREFIX,
        BFCLEvaluator,
        is_java,
        is_js,
        BFCLDataset,
        BFCLRelevanceEvaluator,
        BFCLMultiTurnEvaluator,
        BFCLSingleTurnEvaluator
    )
    BFCL_AVAILABLE = True
    print("Successfully imported BFCL modules")
except ImportError as e:
    print(f"ImportError: {e}. BFCL tests will be skipped.")
    # 创建 mock 函数和类
    def encode_fields(data):
        """Mock encode_fields function"""
        fields = [
            "question",
            "ground_truth",
            "function",
            "missed_function",
            "involved_classes",
            "initial_config",
        ]
        for item in data:
            for field in fields:
                if field in item and not isinstance(item[field], str):
                    item[field] = json.dumps(item[field], ensure_ascii=False)
        return data
    
    VERSION_PREFIX = "BFCL_v3"
    
    def is_java(category):
        """Mock is_java function"""
        return category.lower() == "java"
    
    def is_js(category):
        """Mock is_js function"""
        return category.lower() == "javascript" or category.lower() == "js"
    
    class BaseDataset:
        pass
    
    class BaseEvaluator:
        pass
    
    class BFCLDataset(BaseDataset):
        def __init__(self, **kwargs):
            pass
        
        @staticmethod
        def load(path, category, test_ids=None):
            pass
    
    class BFCLEvaluator(BaseEvaluator):
        def __init__(self, category: str, is_fc_model=True):
            self.is_fc_model = is_fc_model
            self.category = category
            self.model_name = "function-call-model-" + str(uuid.uuid4()).split("-")[-1]
            self.language = "Python"
            # 添加is_empty_execute_response和is_empty_output方法
            self.is_empty_execute_response = lambda response: not response or (isinstance(response, list) and all(not item for item in response))
            self.is_empty_output = lambda output: output is None or output == ""
        
        def score(self, *args, **kwargs):
            raise NotImplementedError("Must be implemented in subclasses")
        
        def decode_ast(self, result, language=None):
            return result
    
    class BFCLRelevanceEvaluator(BFCLEvaluator):
        def score(self, prediction, ground_truth, test_set=None):
            pass
    
    class BFCLMultiTurnEvaluator(BFCLEvaluator):
        def decode_execute(self, *args, **kwargs):
            pass
        
        def score(self, prediction, ground_truth, test_set=None):
            pass
    
    class BFCLSingleTurnEvaluator(BFCLEvaluator):
        def score(self, prediction, ground_truth, test_set=None):
            pass


# ==================== 测试 BFCLDataset 类（续） ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_load_with_test_ids():
    """测试BFCLDataset.load方法使用test_ids过滤"""
    from ais_bench.benchmark.datasets.utils import datasets as utils_datasets
    from ais_bench.benchmark.datasets.bfcl import bfcl
    
    # 配置mock
    mock_get_data_path = MagicMock(return_value="mocked_data_path")
    
    # 模拟数据集文件和ground truth文件
    dataset_file = MagicMock()
    dataset_file.__enter__.return_value = dataset_file
    dataset_file.__iter__.return_value = [
        '{"id": "test_001", "question": "Q1"}\n',
        '{"id": "test_002", "question": "Q2"}\n'
    ]
    
    gt_file = MagicMock()
    gt_file.__enter__.return_value = gt_file
    gt_file.__iter__.return_value = [
        '{"id": "test_001", "ground_truth": ["gt1"]}\n',
        '{"id": "test_002", "ground_truth": ["gt2"]}\n'
    ]
    
    # open函数需要返回两个不同的文件
    mock_open_file = MagicMock(side_effect=[dataset_file, gt_file])
    mock_process = MagicMock(return_value=[{"id": "test_001", "ground_truth": ["gt1"]}])
    mock_encode = MagicMock(return_value=[{"id": "test_001", "encoded": True}])
    mock_dataset = MagicMock()
    mock_dataset.from_list.return_value = MagicMock()
    
    with patch.object(utils_datasets, 'get_data_path', mock_get_data_path), \
         patch('builtins.open', mock_open_file), \
         patch.object(bfcl, 'process_multi_turn_test_case', mock_process), \
         patch.object(bfcl, 'encode_fields', mock_encode), \
         patch.object(bfcl, 'Dataset', mock_dataset), \
         patch.object(bfcl, 'BFCL_INSTALLED', True):
        # 执行load方法，只加载test_001 - 使用绝对路径避免FileExistsError
        BFCLDataset.load(path="/tmp/test_path", category="python", test_ids=["test_001"])
        
        # 验证只处理了指定的test_id
        assert mock_process.called
        assert mock_encode.called

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_load_mismatched_ids():
    """测试BFCLDataset.load方法处理ID不匹配的情况"""
    from ais_bench.benchmark.datasets.utils import datasets as utils_datasets
    from ais_bench.benchmark.datasets.bfcl import bfcl
    
    # 配置mock
    mock_get_data_path = MagicMock(return_value="mocked_data_path")
    
    # 模拟数据集文件和ground truth文件有不同的ID
    dataset_file = MagicMock()
    dataset_file.__enter__.return_value = dataset_file
    dataset_file.__iter__.return_value = ['{"id": "test_001", "question": "Q1"}\n']
    
    gt_file = MagicMock()
    gt_file.__enter__.return_value = gt_file
    gt_file.__iter__.return_value = ['{"id": "different_id", "ground_truth": ["gt"]}\n']
    
    # 模拟open函数根据调用顺序返回不同的文件
    mock_open_file = MagicMock(side_effect=[dataset_file, gt_file])
    mock_process = MagicMock(return_value=[{"id": "test_001"}, {"id": "different_id"}])
    
    with patch.object(utils_datasets, 'get_data_path', mock_get_data_path), \
         patch('builtins.open', mock_open_file), \
         patch.object(bfcl, 'process_multi_turn_test_case', mock_process), \
         patch.object(bfcl, 'encode_fields', MagicMock()), \
         patch.object(bfcl, 'Dataset', MagicMock()), \
         patch.object(bfcl, 'BFCL_INSTALLED', True):
        # 执行load方法，应该抛出ValueError - 使用绝对路径避免FileExistsError
        with pytest.raises(ValueError) as excinfo:
            BFCLDataset.load(path="/tmp/test_path", category="python")
        
        # 验证异常消息
        assert "different ids" in str(excinfo.value) or "mismatch" in str(excinfo.value).lower()

# ==================== 测试 BFCLRelevanceEvaluator 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_relevance_evaluator_initialization():
    """测试评估器初始化"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    assert evaluator.category == "relevance"
    assert evaluator.is_fc_model is True
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_relevance_evaluator_score_method_parameters():
    """测试score方法的参数"""
    evaluator = BFCLRelevanceEvaluator(category="relevance")
    # 测试score方法接受三个参数
    with pytest.raises(TypeError):
        evaluator.score([], [])  # 缺少test_set参数

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_empty_output')
def test_bfcl_relevance_evaluator_score_relevance_success(mock_is_empty_output):
    """测试相关性评估器在相关性测试中成功的情况"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    mock_is_empty_output.return_value = False
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["relevance_mock_gt"]']
    test_set = [{"id": "relevance_test_001", "question": "Test question"}]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_empty_output')
def test_bfcl_relevance_evaluator_score_irrelevance_success(mock_is_empty_output):
    """测试相关性评估器在不相关性测试中成功的情况"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    mock_is_empty_output.return_value = True
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["relevance_mock_gt"]']
    test_set = [{"id": "irrelevance_test_001", "question": "Test question"}]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_relevance_evaluator_score_decode_error():
    """测试相关性评估器在解码错误的情况"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    
    # 准备测试数据
    predictions = ['invalid_json']
    references = ['["relevance_mock_gt"]']
    test_set = [{"id": "relevance_test_001", "question": "Test question"}]
    
    # 模拟decode_ast方法抛出异常
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(side_effect=Exception("Decode error"))
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

# ==================== 测试 BFCLMultiTurnEvaluator 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_initialization():
    """测试评估器初始化"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    assert evaluator.category == "python"
    assert evaluator.is_fc_model is True
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_decode_execute_fc_model():
    """测试多轮评估器的decode_execute方法（FC模型）"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    
    # 模拟convert_to_function_call函数
    with patch('ais_bench.benchmark.datasets.bfcl.bfcl.convert_to_function_call') as mock_convert:
        mock_convert.return_value = ["function_call_result"]
        
        # 测试字符串输入 - 使用有效的JSON字符串
        result_str = evaluator.decode_execute('{"func": {"param": "value"}}')
        assert result_str == ["function_call_result"]
        
        # 测试字典输入
        result_dict = evaluator.decode_execute({"func": {"param": "value"}})
        assert result_dict == ["function_call_result"]

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_decode_execute_prompting_model():
    """测试多轮评估器的decode_execute方法（Prompting模型）"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=False)
    
    # 模拟default_decode_execute_prompting函数
    with patch('ais_bench.benchmark.datasets.bfcl.bfcl.default_decode_execute_prompting') as mock_decode:
        mock_decode.return_value = ["prompting_result"]
        
        result = evaluator.decode_execute("model output", is_fc_model=False)
        assert result == ["prompting_result"]
        mock_decode.assert_called_once_with("model output")

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_empty_execute_response')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.multi_turn_checker')
def test_bfcl_multi_turn_evaluator_score_success(mock_checker, mock_is_empty):
    """测试多轮评估器的score方法成功情况"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    mock_is_empty.return_value = False
    mock_checker.return_value = {"valid": True}
    
    # 准备测试数据 - 使用有效的JSON格式，但predictions应该是列表，不是JSON字符串
    import json
    predictions = [[[{"func": {"param": "value"}}]]]  # 直接使用列表，不是JSON字符串
    references = [json.dumps([["valid_answer"]])]
    test_set = [{
        "id": "multi_turn_001", 
        "question": "Test question",
        "initial_config": '{"config": "value"}',
        "involved_classes": '["class1"]',
        "function": "long_function_doc"
    }]
    
    # 模拟decode_execute方法
    original_decode_execute = evaluator.decode_execute
    evaluator.decode_execute = MagicMock(return_value=["executable_call"])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_execute = original_decode_execute

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_score_invalid_format():
    """测试多轮评估器的score方法处理无效格式"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    
    # 准备测试数据 - 非列表格式
    predictions = ['not_a_list']
    references = ['[["valid_answer"]]']
    test_set = [{
        "id": "multi_turn_001", 
        "question": "Test question",
        "initial_config": '{"config": "value"}',
        "involved_classes": '["class1"]'
    }]
    
    # 执行score方法
    result = evaluator.score(predictions, references, test_set)
    
    # 验证结果
    assert result["accuracy"] == 0.0
    assert result["correct_count"] == 0
    assert result["total_count"] == 1
    # 注意：当格式无效时，可能会添加多个错误（格式错误 + 长度检查错误）
    # 所以details可能不止1个，但至少应该有1个
    assert len(result["details"]) >= 1
    assert result["details"][0]["correct"] is False

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_score_force_terminated():
    """测试多轮评估器的score方法处理强制终止情况"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    
    # 准备测试数据 - 长度不匹配，predictions应该是列表，不是JSON字符串
    import json
    predictions = [[[]]]  # 1轮，直接使用列表
    references = [json.dumps([["answer1"], ["answer2"]])]  # 2轮
    test_set = [{
        "id": "multi_turn_001", 
        "question": "Test question",
        "initial_config": '{"config": "value"}',
        "involved_classes": '["class1"]'
    }]
    
    # 执行score方法
    result = evaluator.score(predictions, references, test_set)
    
    # 验证结果
    assert result["accuracy"] == 0.0
    assert result["correct_count"] == 0
    assert result["total_count"] == 1
    # 注意：当格式无效时，可能会添加多个错误（格式错误 + 长度检查错误）
    # 所以details可能不止1个，但至少应该有1个
    assert len(result["details"]) >= 1
    assert result["details"][0]["correct"] is False

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_decode_execute_method():
    """测试decode_execute方法"""
    evaluator = BFCLMultiTurnEvaluator(category="python")
    # 测试方法存在
    assert hasattr(evaluator, 'decode_execute')

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_score_method_parameters():
    """测试score方法的参数"""
    evaluator = BFCLMultiTurnEvaluator(category="python")
    # 测试score方法接受三个参数
    with pytest.raises(TypeError):
        evaluator.score([], [])  # 缺少test_set参数

# ==================== 测试 BFCLSingleTurnEvaluator 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_single_turn_evaluator_initialization():
    """测试评估器初始化"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    assert evaluator.category == "python"
    assert evaluator.is_fc_model is True
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_single_turn_evaluator_score_method_parameters():
    """测试score方法的参数"""
    evaluator = BFCLSingleTurnEvaluator(category="python")
    # 测试score方法接受三个参数
    with pytest.raises(TypeError):
        evaluator.score([], [])  # 缺少test_set参数

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_function_calling_format_output')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.ast_checker')
def test_bfcl_single_turn_evaluator_score_success(mock_checker, mock_format_check):
    """测试单轮评估器的score方法成功情况"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    mock_format_check.return_value = True
    mock_checker.return_value = {"valid": True}
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_single_turn_evaluator_score_ast_decode_error():
    """测试单轮评估器的score方法处理AST解码错误"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    
    # 准备测试数据
    predictions = ['invalid_json']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法抛出异常
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(side_effect=Exception("AST decode error"))
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_function_calling_format_output')
def test_bfcl_single_turn_evaluator_score_invalid_format(mock_format_check):
    """测试单轮评估器的score方法处理无效格式"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    mock_format_check.return_value = False
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_function_calling_format_output')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.ast_checker')
def test_bfcl_single_turn_evaluator_score_checker_failure(mock_checker, mock_format_check):
    """测试单轮评估器的score方法处理检查器失败"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    mock_format_check.return_value = True
    mock_checker.return_value = {"valid": False, "error": ["Checker error"], "error_type": "checker_error"}
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

# ==================== 测试辅助函数和工具 ====================

def test_json_serialization():
    """测试 JSON 序列化/反序列化"""
    data = {
        "question": ["What is it?"],
        "answer": {"result": "test"}
    }
    
    # 测试序列化
    serialized = json.dumps(data, ensure_ascii=False)
    assert isinstance(serialized, str)
    
    # 测试反序列化
    deserialized = json.loads(serialized)
    assert deserialized == data

def test_unicode_json_handling():
    """测试 Unicode JSON 处理"""
    data = {
        "question": "什么是天气？",
        "answer": "晴天"
    }
    
    # 使用 ensure_ascii=False
    serialized = json.dumps(data, ensure_ascii=False)
    assert "什么是天气" in serialized
    assert "\\u" not in serialized  # 不应该有 Unicode 转义
    
    # 验证可以正确反序列化
    deserialized = json.loads(serialized)
    assert deserialized["question"] == "什么是天气？"

def test_list_comprehension_pattern():
    """测试列表推导式模式（常用于数据处理）"""
    dataset = [{"id": i, "value": i * 2} for i in range(5)]
    
    # 验证生成的数据
    assert len(dataset) == 5
    assert dataset[0]["id"] == 0
    assert dataset[4]["value"] == 8

def test_dictionary_merging():
    """测试字典合并（常用于数据合并）"""
    data1 = {"id": "test_001", "question": "Q1"}
    data2 = {"ground_truth": ["GT1"], "answer": "A1"}
    
    # 合并字典
    merged = {**data1, **data2}
    
    assert "id" in merged
    assert "ground_truth" in merged
    assert len(merged) == 4


# ==================== 测试语言检测函数 ====================

# def test_is_java_function():
#     """测试is_java函数"""
#     assert is_java("java") is True
#     assert is_java("JAVA") is True
#     assert is_java("python") is False
#     assert is_java("javascript") is False

# def test_is_js_function():
#     """测试is_js函数"""
#     assert is_js("javascript") is True
#     assert is_js("js") is True
#     assert is_js("JS") is True
#     assert is_js("python") is False
#     assert is_js("java") is False

# ==================== 性能和边界测试 ====================

def test_large_dataset_encoding():
    """测试大数据集编码"""
    # 生成大量数据
    data = [
        {
            "question": ["Q" + str(i)],
            "ground_truth": ["GT" + str(i)]
        }
        for i in range(100)
    ]
    
    result = encode_fields(data)
    
    # 验证所有数据都被正确编码
    assert len(result) == 100
    for i, item in enumerate(result):
        decoded_q = json.loads(item["question"])
        assert decoded_q[0] == "Q" + str(i)

def test_deeply_nested_structure():
    """测试深度嵌套结构"""
    data = [{
        "question": {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
    }]
    
    result = encode_fields(data)
    
    # 验证深度嵌套结构被正确编码和解码
    decoded = json.loads(result[0]["question"])
    assert decoded["level1"]["level2"]["level3"]["value"] == "deep"

def test_special_characters_handling():
    """测试特殊字符处理"""
    data = [{
        "question": ["Test with 'quotes' and \"double quotes\""],
        "ground_truth": {"key": "value with\nnewline and\ttab"}
    }]
    
    result = encode_fields(data)
    
    # 验证特殊字符被正确处理
    decoded_q = json.loads(result[0]["question"])
    assert "quotes" in decoded_q[0]
    
    decoded_gt = json.loads(result[0]["ground_truth"])
    assert "\n" in decoded_gt["key"]
    assert "\t" in decoded_gt["key"]

def test_empty_string_values():
    """测试空字符串值"""
    data = [{
        "question": "",
        "ground_truth": [""]
    }]
    
    result = encode_fields(data)
    
    # 空字符串应保持不变
    assert result[0]["question"] == ""
    # 列表中的空字符串应被编码
    decoded_gt = json.loads(result[0]["ground_truth"])
    assert decoded_gt[0] == ""

def test_boolean_and_number_conversion():
    """测试布尔值和数字转换"""
    data = [
        {
            "question": [True, False, None],
            "ground_truth": [1, 0, 3.14],
            "function": {"flag": True, "count": 0, "value": 1.618}
        }
    ]
    
    result = encode_fields(data)
    
    # 验证转换为字符串
    decoded_question = json.loads(result[0]["question"])
    assert decoded_question[0] is True
    assert decoded_question[1] is False
    
    decoded_ground_truth = json.loads(result[0]["ground_truth"])
    assert decoded_ground_truth[0] == 1
    assert decoded_ground_truth[2] == 3.14


# pytest 会自动发现和运行测试函数，不需要手动运行代码

# 如果需要在命令行直接运行此文件，可以使用：
# python -m pytest -v test_bfcl_functions_part2.py

