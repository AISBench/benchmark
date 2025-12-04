# 随机合成数据集使用指南

## 一. 应用简介

> 该功能仅支持性能测评相关场景，而不支持精度测评相关场景

该功能旨在不提供真实数据集，而考虑使用构造随机的合成数据集的场景，用于大语言模型推理性能基准测试。

## 二. 使用指导

### 2.1 修改配置文件

配置`ais_bench/benchmark/configs/datasets/synthetic` 目录下的配置模板文件中的`synthetic_config`参数。

当前提供了**两种不同构造类型**的随机数据集配置模板文件，用户可根据自己的需求选择合适的模板文件进行修改：

- **`synthetic_gen_string.py`**: 生成随机长度字符串（模拟真实输入）

    ```python
    synthetic_config = {
        "Type": "string",
        ... # 其他参数
    }
    ```

- **`synthetic_gen_tokenid.py`**: 生成随机token id序列（直接输入编码后的Token）

    ```python
    synthetic_config = {
        "Type": "tokenid",
        ... # 其他参数
    }
    ```

**注意：** [`ais_bench/benchmark/configs/datasets/synthetic/synthetic_gen.py`](../../../ais_bench/benchmark/configs/datasets/synthetic/synthetic_gen.py)模板所使用配置需在[`ais_bench/benchmark/configs/datasets/synthetic`](../../../ais_bench/datasets/synthetic/synthetic_config.py)中进行修改。为了统一数据集配置模式，此方法将在后续版本中弃用。

### 2.2 命令执行

在命令行执行如下命令启动测评:

```bash
ais_bench --models {model_api_file} --datasets synthetic_gen_{string/tokenid} {other_option_args}
```

------

## 三. 参数说明

> 以下仅做配置文件中参数的大概说明，详细取值要求需参考配置文件中预留的注释和使用场景。

### 3.1 公共参数

| 参数名称     | 类型   | 说明                   | 取值范围       |
| ------------ | ------ | ---------------------- | -------------- |
| Type         | string | 数据集类型（必填）     | string/tokenid |
| RequestCount | int    | 生成的请求总数（必填） | [1, 1,048,576] |

------

### 3.2 String类型配置（Type="string"时需配置）

```python
"StringConfig" : {
    "Input" : {          # 输入序列配置
        "Method": str,    # 分布类型：uniform/gaussian/zipf
        "Params": {}      # 对应分布的参数
    },
    "Output" : {         # 输出序列配置（参数同上）
        "Method": str,
        "Params": {}
    }
}
```

#### 输入输出分布参数说明

> **关键规则**
>
> - 所有数值参数的最大值默认应小于 2^20（即 1,048,576）
> - 请求的最大输入/输出长度还会受到服务化配置的限制，具体参考该配置文件中的注释描述

| 分布类型     | 参数       | 类型  | 说明                                          | 取值范围          |
| ------------ | ---------- | ----- | --------------------------------------------- | ----------------- |
| **uniform**  | `MinValue` | int   | 输入/出序列的最小长度                         | [1, 1,048,576]    |
|              | `MaxValue` | int   | 输入/出序列的最大长度（可等于MinValue）       | [≥MinValue]       |
| **gaussian** | `Mean`     | float | 分布中心值（均值）                            | [-3.0e38, 3.0e38] |
|              | `Var`      | float | 方差（控制数据分散程度）                      | [0, 3.0e38]       |
|              | `MinValue` | int   | 输入/出序列的硬性下限                         | [1, 1,048,576]    |
|              | `MaxValue` | int   | 输入/出序列的硬性上限                         | [≥MinValue]       |
| **zipf**     | `Alpha`    | float | 形状参数（值越大分布越均匀）                  | (1.0, 10.0]       |
|              | `MinValue` | int   | 输入/出序列的最小长度                         | [1, 1,048,576]    |
|              | `MaxValue` | int   | 输入/出序列的最大长度（**必须**大于MinValue） | [>MinValue]       |

------

### 3.3 TokenId类型配置（Type="tokenid"时需配置）

```python
"TokenIdConfig" : {
    "RequestSize": int,  # 单请求token数量
    "PrefixLen": int     # 所有请求的公共前缀token数量
}
```

------

## 四. 配置示例

### 4.1 String类型示例

#### 1. Uniform均匀分布

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 1000,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 50, "MaxValue": 500}  # 输入长度50-500
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 20, "MaxValue": 200}  # 输出长度20-200
        }
    }
}
```

**特性**：输入/输出长度在区间内等概率分布，适用于基准性能测试

------

#### 2. Gaussian高斯分布

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 800,
    "StringConfig": {
        "Input": {
            "Method": "gaussian",
            "Params": {
                "Mean": 256,       # 中心值256
                "Var": 10,         # 方差10
                "MinValue": 64,    # 实际范围64-512
                "MaxValue": 512
            }
        },
        "Output": {
            "Method": "gaussian",
            "Params": {
                "Mean": 128,
                "Var": 50,
                "MinValue": 32,
                "MaxValue": 256
            }
        }
    }
}
```

**分布特征**：约95%的输入长度在[236,276]范围内（μ±2σ）

------

#### 3. Zipf齐夫分布

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 1200,
    "StringConfig": {
        "Input": {
            "Method": "zipf",
            "Params": {
                "Alpha": 1.5,      # 强长尾效应
                "MinValue": 10,    # 输入长度范围10-1000
                "MaxValue": 1000
            }
        },
        "Output": {
            "Method": "zipf",
            "Params": {
                "Alpha": 2.0,     # 较平缓分布
                "MinValue": 5,
                "MaxValue": 500
            }
        }
    }
}
```

**典型场景**：模拟真实场景中的请求长尾分布，当Alpha=1.5时，约20%的请求占60%的计算量

------

#### 4. 混合分布配置

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 1500,
    "StringConfig": {
        "Input": {
            "Method": "zipf",    # 输入长尾分布
            "Params": {
                "Alpha": 1.2,
                "MinValue": 10,
                "MaxValue": 2000
            }
        },
        "Output": {
            "Method": "uniform",  # 输出均匀分布
            "Params": {
                "MinValue": 50,
                "MaxValue": 300
            }
        }
    }
}
```

------

### 4.2 TokenId类型示例

#### 长文本压力测试

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 1000,
    "TokenIdConfig": {
        "RequestSize": 2048,   # 单请求2048个token
        "PrefixLen": 0
    }
}
```

#### 短文本性能测试

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 5000,
    "TokenIdConfig": {
        "RequestSize": 128,    # 短文本处理场景
        "PrefixLen": 0
    }
}
```

#### 公共前缀性能测试

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 5000,
    "TokenIdConfig": {
        "RequestSize": 512,    # 单请求512个token
        "PrefixLen": 256      # 所有请求前面256个token相等
    }
}
```

------

## 五. 常见问题

### Q1: 选择分布类型？

- **均匀分布**：适用于压力测试的基线场景
- **高斯分布**：模拟真实场景中的平均请求长度
- **Zipf分布**：生成长尾分布数据（如1%的请求占50%的计算量）
- **分布组合建议**：
  - **压力测试**：Input使用zipf分布，Output使用uniform分布
  - **稳定性测试**：Input/Output均使用gaussian分布

### Q2: 为什么在指定了 *输入长度* 情况下，但性能测评结果矩阵还是会显示非预期值？

- **`tokenid`模式**：发送时会将每条由模型权重词表范围下随机生成的token所组成的、指定长度的prompt重新decode为字符串格式后再发送给服务化，由于每个模型的词表映射可能出现多对一、一对多等情况，故可能出现波动
- **`string`模式**：该模式下的输入长度指代输入的string的长度，而非token的长度
- **预处理阶段**：在使用chat相关的api时，可能会在前后进行一些额外的字符串拼接

### Q3：为什么在String模式下，指定了 *输出长度*，但性能测评结果矩阵还是会显示非预期值？

- **相差很大**：请检查在model api配置文件中是否正确配置`generation_kwargs`中的`ignore_eos`参数（可以使得服务化处理输出时忽略终止符直至输出长度达到预设值）为`True`

------

## 六. 注意事项

1. **`tokenid`模式**：该模式下的`tokenid`取值范围取决于在模型配置文件中指定的模型的词表范围

2. **`string`模式**：当MinValue=MaxValue时生成固定长度序列
