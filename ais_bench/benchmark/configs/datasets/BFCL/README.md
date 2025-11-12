# BFCL (Berkeley Function Calling Leaderboard) V3
中文 | [English](README_en.md)
## 数据集简介

**Berkeley Function Calling Leaderboard (BFCL)** 是一个专门用于评估大语言模型（LLMs）函数调用能力的综合性、可执行的评估基准。

### 主要特点

- **首个综合性评估基准**：BFCL是第一个专门针对大语言模型函数调用能力的全面评估平台
- **可执行性验证**：与以往的评估方法不同，BFCL不仅评估模型生成函数调用的能力，还能实际执行这些函数调用，验证其正确性
- **多样化的函数调用形式**：该数据集涵盖了各种不同形式的函数调用，能够全面测试模型在不同场景下的表现
- **丰富的应用场景**：BFCL包含了多种不同的使用场景，确保评估的全面性和实用性

> 🔗 **官方主页**：[https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)

## 数据集部署

BFCL数据集通过Python依赖包的方式集成，数据文件包含在 `bfcl-eval` 依赖包中，安装依赖后即可直接使用，无需额外下载或网络连接。

### 环境要求
- **bfcl-eval** 依赖包（包含完整数据集）

### 安装步骤
```bash
pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps
```

✅ **安装完成后，BFCL数据集已随依赖包一同安装到本地环境中，可在离线环境下正常使用。**

### ⚠️ 使用须知

> - **模型限制**：BFCL数据集只支持使用API模型 [`vllm_api_function_call_chat`](../../models/vllm_api/vllm_api_function_call_chat.py) 进行测评
> - **数据集合并**：BFCL每个子类别数据集采用不同的方式进行eval，因此不支持使用 `--merge-ds` 参数来合并子数据集进行测试

### 使用示例

#### 1. 模型配置
配置 [`vllm_api_function_call_chat`](../../models/vllm_api/vllm_api_function_call_chat.py) 模型：

```python
from ais_bench.benchmark.models import VLLMFunctionCallAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMFunctionCallAPIChat,
        abbr="vllm-api-function-call-chat",
        path="",
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",        # 推理服务IP地址
        host_port=8080,             # 推理服务端口号
        max_out_len=10240,          # 最大输出token长度
        batch_size=100,             # 并发请求批次大小
        returns_tool_calls=True,    # 函数调用信息提取方式（支持tool_calls字段时设为True）
        trust_remote_code=False,
        generation_kwargs=dict(     # 生成参数配置
            temperature=0.01,       # 建议使用低温度值以减少精度波动
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

#### 2. 执行测评
运行AISBench测评命令：

```bash
# 基本命令格式
ais_bench --models vllm_api_function_call_chat --datasets {BFCL数据集配置}

# 具体示例：测试简单函数调用
ais_bench --models vllm_api_function_call_chat --datasets BFCL_gen_simple
```

#### 3. 结果示例
测评完成后的精度结果展示：

```bash
dataset          version    metric    mode      vllm-api-function-call-chat
---------       ---------  --------  ------  --------------------------------
BFCL-v3-simple   542b40    accuracy   gen            0.96 (385/400)
```

## 数据集分类

通过 `--datasets` 参数可以灵活选择测试范围，支持以下三种粒度的测试配置：

### 单独测试类别

- [`BFCL_gen_simple`](./BFCL_gen_simple.py) - 简单 Python 函数调用
- [`BFCL_gen_java`](./BFCL_gen_java.py) - 简单 Java 函数调用
- [`BFCL_gen_javascript`](./BFCL_gen_javascript.py) - 简单 JavaScript 函数调用
- [`BFCL_gen_parallel`](./BFCL_gen_parallel.py) - 并行函数调用
- [`BFCL_gen_multiple`](./BFCL_gen_multiple.py) - 多函数顺序调用
- [`BFCL_gen_parallel_multiple`](./BFCL_gen_parallel_multiple.py) - 并行与顺序混合调用
- [`BFCL_gen_irrelevance`](./BFCL_gen_irrelevance.py) - 含无关文档的函数调用
- [`BFCL_gen_live_simple`](./BFCL_gen_live_simple.py) - 实时：简单函数调用
- [`BFCL_gen_live_multiple`](./BFCL_gen_live_multiple.py) - 实时：多函数顺序调用
- [`BFCL_gen_live_parallel`](./BFCL_gen_live_parallel.py) - 实时：并行函数调用
- [`BFCL_gen_live_parallel_multiple`](./BFCL_gen_live_parallel_multiple.py) - 实时：并行与顺序混合调用
- [`BFCL_gen_live_irrelevance`](./BFCL_gen_live_irrelevance.py) - 实时：含无关文档的函数调用
- [`BFCL_gen_live_relevance`](./BFCL_gen_live_relevance.py) - 实时：含相关文档的函数调用
- [`BFCL_gen_multi_turn_base`](./BFCL_gen_multi_turn_base.py) - 多轮：基础场景
- [`BFCL_gen_multi_turn_miss_func`](./BFCL_gen_multi_turn_miss_func.py) - 多轮：缺失函数
- [`BFCL_gen_multi_turn_miss_param`](./BFCL_gen_multi_turn_miss_param.py) - 多轮：缺失参数
- [`BFCL_gen_multi_turn_long_context`](./BFCL_gen_multi_turn_long_context.py) - 多轮：长上下文

### 测试组别
适用于批量测试，一次性运行多个相关测试类别：

- [`BFCL_gen_all`](./BFCL_gen_all.py) - 全量测试：包含所有测试类别
- [`BFCL_gen_single_turn`](./BFCL_gen_single_turn.py) - 单轮对话测试：
  - [`BFCL_gen_simple`](./BFCL_gen_simple.py)：简单函数调用（单轮：基础 Python 函数调用）
  - [`BFCL_gen_irrelevance`](./BFCL_gen_irrelevance.py)：含无关文档（单轮：含无关文档的函数调用）
  - [`BFCL_gen_parallel`](./BFCL_gen_parallel.py)：并行调用（单轮：并行函数调用）
  - [`BFCL_gen_multiple`](./BFCL_gen_multiple.py)：多函数顺序调用（单轮：多函数顺序调用）
  - [`BFCL_gen_parallel_multiple`](./BFCL_gen_parallel_multiple.py)：并行与顺序混合（单轮：并行与顺序混合调用）
  - [`BFCL_gen_java`](./BFCL_gen_java.py)：Java 函数调用（单轮：基础 Java 函数调用）
  - [`BFCL_gen_javascript`](./BFCL_gen_javascript.py)：JavaScript 函数调用（单轮：基础 JavaScript 函数调用）
- [`BFCL_gen_multi_turn`](./BFCL_gen_multi_turn.py) - 多轮对话测试：
  - [`BFCL_gen_multi_turn_base`](./BFCL_gen_multi_turn_base.py)：多轮：基础场景
  - [`BFCL_gen_multi_turn_miss_func`](./BFCL_gen_multi_turn_miss_func.py)：多轮：缺失函数
  - [`BFCL_gen_multi_turn_miss_param`](./BFCL_gen_multi_turn_miss_param.py)：多轮：缺失参数
  - [`BFCL_gen_multi_turn_long_context`](./BFCL_gen_multi_turn_long_context.py)：多轮：长上下文
- [`BFCL_gen_live`](./BFCL_gen_live.py) - 实时测试：
  - [`BFCL_gen_live_simple`](./BFCL_gen_live_simple.py)：实时-简单调用
  - [`BFCL_gen_live_multiple`](./BFCL_gen_live_multiple.py)：实时-多函数顺序
  - [`BFCL_gen_live_parallel`](./BFCL_gen_live_parallel.py)：实时-并行调用
  - [`BFCL_gen_live_parallel_multiple`](./BFCL_gen_live_parallel_multiple.py)：实时-并行与顺序混合
  - [`BFCL_gen_live_irrelevance`](./BFCL_gen_live_irrelevance.py)：实时-含无关文档
  - [`BFCL_gen_live_relevance`](./BFCL_gen_live_relevance.py)：实时-含相关文档
- [`BFCL_gen_non_live`](./BFCL_gen_non_live.py) - 标准测试：
  - [`BFCL_gen_simple`](./BFCL_gen_simple.py)：简单函数调用
  - [`BFCL_gen_irrelevance`](./BFCL_gen_irrelevance.py)：含无关文档
  - [`BFCL_gen_parallel`](./BFCL_gen_parallel.py)：并行调用
  - [`BFCL_gen_multiple`](./BFCL_gen_multiple.py)：多函数顺序调用
  - [`BFCL_gen_parallel_multiple`](./BFCL_gen_parallel_multiple.py)：并行与顺序混合
  - [`BFCL_gen_java`](./BFCL_gen_java.py)：Java 函数调用
  - [`BFCL_gen_javascript`](./BFCL_gen_javascript.py)：JavaScript 函数调用

### 精确测试配置
适用于调试和精确验证特定测试用例：

使用 `--datasets BFCL_gen_ids` 可以指定具体的测试用例ID进行精确测试。

#### 配置方法
编辑 [`BFCL_gen_ids.py`](./BFCL_gen_ids.py) 文件中的 `test_ids_to_generate` 字典：

```python
test_ids_to_generate = {
    "simple": ["simple_0"], # 指定BFCL_gen_simple中名称为simple_0的case
    "irrelevance": [],      # 空list表示不指定该类别的case
    "parallel": ["parallel_0"],
    "multiple": ["multiple_0"],
    "parallel_multiple": ["parallel_multiple_0"],
    "java": [],
    "javascript": ["javascript_0"],
    "live_simple": ["live_simple_0-0-0"],
    "live_multiple": ["live_multiple_0-0-0"],
    "live_parallel": ["live_parallel_0-0-0"],
    "live_parallel_multiple": ["live_parallel_multiple_0-0-0"],
    "live_irrelevance": ["live_irrelevance_0-0-0"],
    "live_relevance": ["live_relevance_0-0-0"],
    "multi_turn_base": ["multi_turn_base_0"],
    "multi_turn_miss_func": ["multi_turn_miss_func_0"],
    "multi_turn_miss_param": ["multi_turn_miss_param_0"],
    "multi_turn_long_context": ["multi_turn_long_context_0"],
}
```

## 💡 使用建议

1. **首次评估**：建议使用 [测试组别](#测试组别) 进行相对全面的评估
2. **性能调优**：根据初步结果选择特定类别进行深入测试
3. **问题定位**：使用精确测试配置定位具体问题用例

---

