# τ²-Bench
## τ²-Bench Evaluation Benchmark Introduction
**τ²-Bench (Tau Squared Bench, also written as TAU2-Bench)** is an authoritative large model agent evaluation benchmark launched by Princeton University and Sierra Research in 2025. It primarily evaluates **Dual-Control** environment's dialogue, tool calling, and compliance capabilities.
Official benchmark repository: [https://github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

### 1. Core Positioning and Background
- **Predecessor**: Enhanced version of τ-Bench, fixing numerous code issues and adding new domains.
- **Core Innovation**: **Dual-Control Interaction**
  - AI agents **share control with users** over a shared dynamic environment (not one-way instructions).
  - Both parties can call tools, modify states, and verify results, closer to real collaboration scenarios.
- **Core Objective**: Evaluate agents' comprehensive capabilities in **task completion, tool usage, policy compliance, and multi-turn communication**.

### 2. Three Evaluation Domains
Covering real customer service scenarios, each domain includes independent **business policies, tool sets, databases, and task sets**:

1. **Airline**
   - Tasks: Flight inquiry, rescheduling, refund, seat selection, baggage, mileage redemption.
   - Tools: Ticket check, rescheduling, cancellation, payment, membership inquiry.

2. **Retail (E-commerce)**
   - Tasks: Order inquiry, modification, refund, exchange, complaint, coupon.
   - Tools: Order check, return, exchange, replacement, discount.

3. **Telecom (Newly Added)**
   - Tasks: Plan application, data/phone bill inquiry, network troubleshooting, suspension/activation, international roaming.
   - Tools: Account opening, suspension, bill check, troubleshooting, roaming activation.

### 3. Core Evaluation Mechanism (Fine-grained Assessment)
Adopts **5 automatic verification methods** to avoid coarse-grained issues of single keyword matching:
- **db_check**: Whether the final database state meets expectations (e.g., roaming enabled).
- **tool_call**: Whether tool calling sequence, parameters, and frequency are correct.
- **policy_check**: Whether business rules are violated (e.g., excessive refunds, prohibited items).
- **goal_check**: Whether user core needs are met.
- **dialogue_check**: Whether responses are accurate, compliant, and natural.

### 4. Key Evaluation Metrics
- **Pass^1**: Single-round task **complete success rate** (primary metric).
- **Pass^k**: Probability of all successes in k independent runs (stability).
- **Error decomposition**:
  - **Reasoning errors**: Logic, tool selection, policy understanding errors.
  - **Interaction errors**: Communication ambiguity, information missing, user collaboration failure.
- **Domain average score**: Comprehensive score across the three domains (core of leaderboard).

### 5. Technical Features
- **Dual-control modeling**: Formulated as **Dec-POMDP** (Decentralized Partially Observable Markov Decision).
- **Reliable user simulator**: AI simulates users, strongly coupled with environment, reducing simulation bias.
- **Automatic evaluation**: No human intervention throughout, reproducible and comparable.
- **Multi-mode evaluation**:
  - **Standard mode**: Dual-control collaboration (real scenario).
  - **No-User mode**: Agent full control (isolated reasoning ability).

## Quick Start with τ²-Bench Evaluation in AISBench
### 1. Prepare Inference Services
Ensure local or cloud deployment of tested inference services following OpenAI chat/completions API specification with tool call support, and inference services for simulating users (e.g., VLLM, OpenAI, etc.).

### 2. **Install AISBench Evaluation Tool & τ²-Bench Additional Dependencies**
1. Refer to [AISBench Installation Documentation](../../get_started/install.md) to install AISBench evaluation tool.
2. Install τ²-Bench additional dependencies:
   ```bash
   # Execute in AISBench tool root directory
   pip install -r requirements/datasets/tau2_dependencies.txt
   ```
> Note: The commit id of the tau2-bench repository used by AISBench is c5b2d228d850c59b749b93cf32c4745d3aa53967 (version from February 2025).

### 3. Configure Custom Configuration File for τ²-Bench Tasks
1. Modify necessary configurations in `ais_bench/configs/agent_examples/tau2_bench_task.py` under AISBench tool root directory (mainly configuring information about tested inference services and user-simulating inference services)
```python
# ......
models = [
    dict(
        abbr="openai-v1-chat",
        api_key=None, # API KEY default is an invalid string, OPENAI_API_KEY will be declared internally
        agent = None,                 # Agent implementation used, default is DEFAULT_AGENT_IMPLEMENTATION
        llm_agent = "openai/qwen3",               # Required, LLM used by agent, fill in "openai/{model name of inference service}"
        llm_args_agent = { # Parameters for agent LLM, support passing other parameters compatible with openai interface format
            "api_base": "http://localhost:2498/v1", # Required, base_url of inference service
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
                domain = task,                      # -d, simulation domain to run, optional values: "airline", "retail", "telecom"
                num_trials = 1,                     # Number of runs per task, default is 1
                user = None,                  # User implementation used, default is DEFAULT_USER_IMPLEMENTATION
                llm_user = "openai/qwen3",                # Required, LLM used by user, fill in "openai/{model name of inference service}"
                llm_args_user = { # Parameters for user LLM, support passing other parameters compatible with openai interface format
                    "api_base": "http://localhost:2498/v1", # Required, base_url of inference service
                    "temperature": 0.0
                },
                # ......
                max_concurrency = 5,               # Maximum concurrency for a single task, default is DEFAULT_MAX_CONCURRENCY=5
            ),
        )
    )
# ......
```
- The max_concurrency in the configuration file represents the maximum concurrency for a single task ("airline", "retail", "telecom"), default value is 5.

### 4. Execute τ²-Bench Tasks
1. Execute the following command in AISBench tool root directory:
   ```bash
   # Execute τ²-Bench tasks
   ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3
   ```
   - The `--max-num-workers` parameter indicates the maximum task concurrency. `--max-num-workers 3` means the three tasks "airline", "retail", "telecom" will be executed in parallel.

2. Execution process dashboard example

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
During execution, all result files will be generated in the `outputs/default/{timestamp}` (e.g., `outputs/default/20260408_091146`) directory. During the process, you can view the detailed execution logs of the corresponding tasks in `outputs/default/{timestamp}/logs/eval/openai-v1-chat/tau2_bench_{task name}.out`.

3. After task execution is complete, the following accuracy results will be printed:
```shell
| dataset | version | metric | mode | total_count | openai-v1-chat |
|----- | ----- | ----- | ----- | ----- | -----|
| tau2_bench_airline | / | pass^1 | unknown | 50 | 38.00 |
| tau2_bench_retail | / | pass^1 | unknown | 114 | 21.05 |
| tau2_bench_telecom | / | pass^1 | unknown | 114 | 33.33 |
| tau2_bench_avg | - | naive_average | unknown | / | 30.80 |
| tau2_bench_avg-weighted | - | weighted_average | unknown | / | 29.14 |
```
- `tau2_bench_avg` represents the simple average score across the three domains.
- `tau2_bench_avg-weighted` represents the weighted average score across the three domains (weights are the number of tasks in each domain).

4. The structure of result files in the final `outputs/default/{timestamp}` directory is as follows:

```shell
outputs/default/20260408_091146
├── configs
│   └── 20260409_111604_1191827.py
├── logs # Process logs
│   └── eval
│       └── openai-v1-chat
│           ├── tau2_bench_airline.out # Detailed execution logs for airline evaluation
│           ├── tau2_bench_retail.out # Detailed execution logs for retail evaluation
│           └── tau2_bench_telecom.out # Detailed execution logs for telecom evaluation
├── results # Final results
│   └── openai-v1-chat
│       ├── tau2_bench_airline # Detailed execution results for airline evaluation
│       │   └── tau2_run_detail.json
│       ├── tau2_bench_airline.json # Accuracy results for airline tasks
│       ├── tau2_bench_retail # Detailed execution results for retail tasks
│       │   └── tau2_run_detail.json
│       ├── tau2_bench_retail.json # Accuracy results for retail tasks
│       ├── tau2_bench_telecom # Detailed execution results for telecom tasks
│       │   └── tau2_run_detail.json
│       └── tau2_bench_telecom.json # Accuracy results for telecom tasks
└── summary # Final aggregated accuracy results
    ├── summary_20260409_111604.csv
    ├── summary_20260409_111604.md
    └── summary_20260409_111604.txt

```


## Continue Evaluation After Interruption
In cases of high concurrency, some model services may return errors during extensive multi-turn conversations, causing task failures, for example:
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

At this point, you can manually interrupt task execution, for example by pressing `Ctrl + C`, and then execute the following command to continue evaluation based on the previously completed evaluation progress:
```bash
# ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3 --reuse {timestamp}
ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3 --reuse 20260408_091146
```

## Multiple Executions of a Single Case (pass^k)
1. Modify the value of the `num_trials` parameter in `ais_bench/configs/agent_examples/tau2_bench_task.py` under AISBench tool root directory to the number of executions needed (default is 1)
```python
# ......
sub_tasks = ["airline", "retail", "telecom"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'tau2_bench_{task}',
            args = dict(
                domain = task,                      # -d, simulation domain to run, optional values: "airline", "retail", "telecom"
                num_trials = 3,                     # Number of runs per task, default is 1
                # ......
            ),
        )
    )
# ......
```

2. After executing the `ais_bench ais_bench/configs/agent_examples/tau2_bench_task.py --max-num-workers 3` command, each case will be executed `num_trials` times, and the total number in the progress bar will also increase to `num_trials` times.
```
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| Task Name                         |   Process | Progress                                                   | Time Cost   | Status   | Log Path                                        | Extend Parameters   |
+===================================+===========+============================================================+=============+==========+=================================================+=====================+
| openai-v1-chat/tau2_bench_airline |   1856223 | [######                        ] 30/150 Running TAU2 Bench | 0:07:13     | running  | logs/eval/openai-v1-chat/tau2_bench_airline.out | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| openai-v1-chat/tau2_bench_retail  |   1856224 | [######                        ] 75/342 Running TAU2 Bench | 0:11:56     | running  | logs/eval/openai-v1-chat/tau2_bench_retail.out  | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| openai-v1-chat/tau2_bench_telecom |   1856222 | [######                        ] 76/342 Running TAU2 Bench | 1:09:51     | running  | logs/eval/openai-v1-chat/tau2_bench_telecom.out | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
```

3. The final printed accuracy results are as follows:
```shell
| dataset | version | metric | mode | total_count | openai-v1-chat |
|----- | ----- | ----- | ----- | ----- | -----|
| tau2_bench_airline | / | pass^3 | unknown | 50 | 38.00 |
| tau2_bench_retail | / | pass^3 | unknown | 114 | 21.05 |
| tau2_bench_telecom | / | pass^3 | unknown | 114 | 33.33 |
| tau2_bench_avg | - | naive_average | unknown | / | 30.80 |
| tau2_bench_avg-weighted | - | weighted_average | unknown | / | 29.14 |
```