# τ²-Bench
## τ²-Bench 测评基准简介
**τ²-Bench（Tau Squared Bench，也写作TAU2-Bench）** 是由普林斯顿大学与 Sierra Research 于 2025 年推出的权威大模型智能体（Agent）评测基准，核心评估**双控环境（Dual-Control）**下的对话、工具调用与合规能力。
基准官方仓库：[https://github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

### 一、核心定位与背景
- **前身**：τ-Bench 的增强版，修复大量代码问题并新增领域。
- **核心创新**：**双控交互（Dual-Control）**
  - AI 智能体 **与用户共同控制** 共享动态环境（非单向指令）。
  - 双方均可调用工具、修改状态、验证结果，更贴近真实协作场景。
- **核心目标**：评测智能体的 **任务完成、工具使用、策略遵守、多轮沟通** 综合能力。

### 二、四大评测领域
覆盖真实客服场景，每个领域含独立 **业务策略、工具集、数据库、任务集**：

1. **Airline（航空）**
   - 任务：航班查询、改签、退票、选座、行李、里程兑换。
   - 工具：查票、改期、取消、支付、会员查询。

2. **Retail（零售电商）**
   - 任务：订单查询、修改、退款、换货、投诉、优惠券。
   - 工具：查单、退货、换货、补发、折扣。

3. **Telecom（电信，新增）**
   - 任务：套餐办理、流量/话费查询、网络排障、停机复机、国际漫游。
   - 工具：开户、停机、查账单、修障、开通漫游。

### 三、核心评测机制（细粒度评估）
采用 **5 种自动校验方式** 组合，避免单一关键词匹配的粗粒度问题：
- **db_check**：数据库最终状态是否符合预期（如漫游已开启）。
- **tool_call**：工具调用顺序、参数、次数是否正确。
- **policy_check**：是否违反业务规则（如超额退票、禁售品）。
- **goal_check**：用户核心需求是否达成。
- **dialogue_check**：回复是否准确、合规、自然。

### 四、关键评测指标
- **Pass^1**：单轮任务 **完全成功率**（主指标）。
- **Pass^k**：k 轮独立运行中全部成功的概率（稳定性）。
- **错误分解**：
  - **推理错误**：逻辑、工具选择、策略理解错误。
  - **交互错误**：沟通歧义、信息缺失、用户协作失败。
- **领域平均分**：三大领域综合得分（排行榜核心）。

### 五、技术特点
- **双控建模**：形式化为 **Dec-POMDP**（分散式部分可观测马尔可夫决策）。
- **可靠用户模拟器**：AI 模拟用户，与环境强耦合，减少模拟偏差。
- **自动评估**：全程无需人工，可复现、可对比。
- **多模式评估**：
  - **标准模式**：双控协作（真实场景）。
  - **No-User 模式**：Agent 全权控制（隔离推理能力）。

## AISBench中快速上手 τ²-Bench 测评
### 1. 准备推理服务
确保本地或云端部署了遵循 OpenAI chat/completions API 规范且支持tool call的被测推理服务和模拟用户的推理服务（如 VLLM、OpenAI 等）。

### 2. **安装AISBench测评工具&τ²-Bench额外依赖**
1. 参考[AISBench安装文档](../../get_started/install.md)安装AISBench测评工具。
2. 安装τ²-Bench额外依赖：
   ```bash
   # AISBench工具根目录下执行
   pip install -r requirements/datasets/tau2_dependencies.txt
   ```
> 注意，AISBench依赖的tau2-bench仓库的commit id为 c5b2d228d850c59b749b93cf32c4745d3aa53967（2025年2月的版本）。

### 3. 配置τ²-Bench任务的自定义配置文件
1. 在AISBench工具根目录下修改`ais_bench/configs/agent_examples/tau2_bench_task.py`中必要的配置（主要是配置被测推理服务和模拟用户的推理服务的信息）
```python
# ......
models = [
    dict(
        abbr="openai-v1-chat",
        api_key=None, # API KEY 默认是个无效字符串 ,内部会声明OPENAI_API_KEY
        agent = None,                 # 使用的 agent 实现，默认为 DEFAULT_AGENT_IMPLEMENTATION
        llm_agent = "openai/qwen3",               # 必填，agent 使用的 LLM，填写"openai/{推理服务的模型名称}"
        llm_args_agent = { # agent LLM 的参数，支持传其他兼容openai接口格式的参数
            "api_base": "http://localhost:2498/v1", # 必填，推理服务的base_url
            "temperature": 0.5
        },
    )
]

# ......

sub_tasks = ["airline", "retail", "telecom"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'tau2_bench_{task}',
            args = dict(
                domain = task,                      # -d, 要运行的模拟域，可选值为 "airline", "retail", "telecom"
                num_trials = 1,                     # 每个任务运行的次数，默认为 1
                user = None,                  # 使用的 user 实现，默认为 DEFAULT_USER_IMPLEMENTATION
                llm_user = "openai/qwen3",                # 必填，user 使用的 LLM，填写"openai/{推理服务的模型名称}"
                llm_args_user = { # user LLM 的参数，支持传其他兼容openai接口格式的参数
                    "api_base": "http://localhost:2498/v1", # 必填，推理服务的base_url
                    "temperature": 0.0
                },
                # ......
                max_concurrency = 5,               # 并发运行的最大模拟数，默认为 DEFAULT_MAX_CONCURRENCY=5
            ),
        )
    )
# ......
```
- 配置文件中max_concurrency表示单个任务（"airline", "retail", "telecom"）的最大并发数，默认值为5。

### 4. 执行τ²-Bench任务
1. 在AISBench工具根目录下执行以下命令：
   ```bash
   # 执行τ²-Bench任务
   ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3
   ```
   - `--max-num-workers`参数表示任务最大并发数，`--max-num-workers 3`表示"airline", "retail", "telecom"这3个任务会并行执行。

2. 执行过程看板示例

```
Base path of result&log : outputs/default/20260408_091146
Task Progress Table (Updated at: 2026-04-08 10:22:37)
Page: 1/1  Total 4 rows of data
Press Up/Down arrow to page, 'P' to PAUSE/RESUME screen refresh, 'Ctrl + C' to exit

+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| Task Name                         |   Process | Progress                                                   | Time Cost   | Status   | Log Path                                        | Extend Parameters   |
+===================================+===========+============================================================+=============+==========+=================================================+=====================+
| openai-v1-chat/tau2_bench_airline |   1856223 | [######                        ] 10/50 Running TAU2 Bench  | 0:07:13     | running  | logs/eval/openai-v1-chat/tau2_bench_airline.out | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| openai-v1-chat/tau2_bench_retail  |   1856224 | [######                        ] 25/114 Running TAU2 Bench | 0:11:56     | running  | logs/eval/openai-v1-chat/tau2_bench_retail.out  | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| openai-v1-chat/tau2_bench_telecom |   1856222 | [##################            ] 71/114 Running TAU2 Bench | 1:09:51     | running  | logs/eval/openai-v1-chat/tau2_bench_telecom.out | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+

```
执行过程中所有结果文件会生成在`outputs/default/{时间戳}`（如`outputs/default/20260408_091146`）目录下，过程中可以在`outputs/default/{时间戳}/logs/eval/openai-v1-chat/tau2_bench_{任务名称}.out`查看对应任务的详细执行日志。

3. 任务执行完成后，会打印如下精度结果：


4. 最终`outputs/default/{时间戳}`目录下结果文件的结构如下：


## 中断后继续执行测评
并发交大的情况下，部分模型服务在大量多轮对话过程中可能会出现错误返回，导致任务失败，例如：
```
Base path of result&log : outputs/default/20260408_091146
Task Progress Table (Updated at: 2026-04-08 10:22:37)
Page: 1/1  Total 4 rows of data
Press Up/Down arrow to page, 'P' to PAUSE/RESUME screen refresh, 'Ctrl + C' to exit

+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| Task Name                         |   Process | Progress                                                   | Time Cost   | Status   | Log Path                                        | Extend Parameters   |
+===================================+===========+============================================================+=============+==========+=================================================+=====================+
| openai-v1-chat/tau2_bench_airline |   1856223 | [######                        ] 10/50 Running TAU2 Bench  | 0:07:13     | error    | logs/eval/openai-v1-chat/tau2_bench_airline.out | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| openai-v1-chat/tau2_bench_retail  |   1856224 | [######                        ] 25/114 Running TAU2 Bench | 0:11:56     | error    | logs/eval/openai-v1-chat/tau2_bench_retail.out  | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| openai-v1-chat/tau2_bench_telecom |   1856222 | [##################            ] 71/114 Running TAU2 Bench | 1:09:51     | running  | logs/eval/openai-v1-chat/tau2_bench_telecom.out | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
```

这个时候可以手动中断任务执行，例如按下`Ctrl + C`, 随后执行如下命令在之前完成的测评进度基础上继续测评：
```bash
# ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3 --reuse {时间戳}
ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3 --reuse 20260408_091146
```

## 单条case多次执行（pass^k）
1. 在AISBench工具根目录下修改`ais_bench/configs/agent_examples/tau2_bench_task.py` `num_trials`参数的取值为需要执行的次数（默认为1）
```python
# ......
sub_tasks = ["airline", "retail", "telecom"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'tau2_bench_{task}',
            args = dict(
                domain = task,                      # -d, 要运行的模拟域，可选值为 "airline", "retail", "telecom"
                num_trials = 3,                     # 每个任务运行的次数，默认为 1
                # ......
            ),
        )
    )
# ......
```

2. 执行`ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3`命令后每条case会执行`num_trials`次，进度条的总数也会相应增加至`num_trials`倍。

3. 最终打印的精度结果如下：

