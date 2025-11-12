# 性能测评结果说明
性能测评结果包括单个推理请求性能输出结果和端到端性能输出结果，参数说明如下：

## 单个推理请求性能输出结果
部分统计指标解释如下所示：
+ P75 / P90 / P99：以 TPOT 为例，表示所有请求的 TPOT 值分别处于第 75、90、99 百分位的性能表现。
+ E2EL（End-to-End Latency）：单个请求从发送到接收全部响应的总时延。
+ TTFT（Time To First Token）：首个 Token 返回的时延。
+ TPOT（Time Per Output Token）：输出阶段每个 Token 的平均生成时延（不含首个 Token）。
+ ITL（Inter-token Latency）：相邻 Token 间的平均间隔时延（不含首个 Token）。
+ InputTokens：请求的输入 Token 数量。
+ OutputTokens：请求生成的输出 Token 数量。
+ OutputTokenThroughput：输出 Token 的吞吐率（Token/s）。
+ Tokenizer：Tokenizer 编码耗时。
+ Detokenizer：Detokenizer 解码耗时。

|Performance Parameters|Stage|Average|Max|Min|Median|P75|P90|P99|N|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|E2EL|统计此参数的阶段|平均请求时延|最大请求时延|最小请求时延|请求时延中位数|请求时延75分位值|请求时延90分位值|请求时延99分位值|测试数据量，来源于输入参数|
|TTFT|统计此参数的阶段|首个token平均时延|首个token最大时延|首个token最小时延|首个token中位数时延|首个token75分位时延|首个token90分位时延|首个token99分位时延|测试数据量，来源于输入参数|
|TPOT|统计此参数的阶段|Decode阶段平均时延|最大Decode阶段时延|最小Decode阶段时延|Decode阶段中位数时延|75分位Decode阶段时延|90分位每条请求Decode阶段平均时延|99分位Decode阶段时延|测试数据量，来源于输入参数|
|ITL|统计此参数的阶段|token间平均时延|token间最大时延|token间最小时延|token间中位数时延|token间75分位时延|token间90分位时延|token间99分位时延|测试数据量，来源于输入参数|
|InputTokens|统计此参数的阶段|输入token平均长度|最大输入token长度|最小输入token长度|输入token中位数长度|75分位输入token长度|90分位输入token长度|99分位输入token长度|测试数据量，来源于输入参数|
|OutputTokens|统计此参数的阶段|输出token平均长度|最大输出token长度|最小输出token长度|输出token中位数长度|75分位输出token长度|90分位输出token长度|99分位输出token长度|测试数据量，来源于输入参数|
|OutputTokenThroughput|统计此参数的阶段|平均输出吞吐|最大输出吞吐|最小输出吞吐|中位数输出吞吐|输出吞吐75分位|输出吞吐90分位|输出吞吐99分位|测试数据量，来源于输入参数|

## 端到端性能输出结果
| 参数                           | 说明                    |
| ---------------------------- | ---------------------  |
| **Benchmark Duration**       | 测试任务的总执行时间            |
| **Total Requests**           | 请求总数量                 |
| **Failed Requests**          | 请求失败数量（包含无响应或响应为空）    |
| **Success Requests**         | 成功返回的请求数量（包括空响应与非空响应） |
| **Concurrency**              | 实际平均并发数               |
| **Max Concurrency**          | 配置的最大并发数              |
| **Request Throughput**       | 请求级吞吐率（请求数/秒）         |
| **Total Input Tokens**       | 所有请求的总输入 Token 数      |
| **Prefill Token Throughput** | Prefill 阶段的 Token 吞吐率 |
| **Total Output Tokens**      | 所有请求生成的总输出 Token 数    |
| **Input Token Throughput**   | 输入 Token 吞吐率          |
| **Output Token Throughput**  | 输出 Token 吞吐率          |
| **Total Token Throughput**   | 总 Token 吞吐率（输入 + 输出）  |
