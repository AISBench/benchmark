# 合成随机数据集配置文件使用指南

## 一. 应用场景

本配置文件用于生成两种类型的合成数据集，用于大语言模型推理性能基准测试：

- **String类型**: 生成随机长度字符串（模拟真实输入）
- **TokenId类型**: 生成随机token id序列（直接输入编码后的Token）

------

## 二. 参数说明

> [NOTE]
>
> 以下仅做[synthetic_config.py](../../ais_bench/datasets/synthetic/synthetic_config.py)配置文件中参数的大概说明，详细取值要求需参考配置文件中预留的注释和使用场景。

### 2.1 公共参数

| 参数名称     | 类型   | 说明                   | 取值范围       |
| ------------ | ------ | ---------------------- | -------------- |
| Type         | string | 数据集类型（必填）     | string/tokenid |
| RequestCount | int    | 生成的请求总数（必填） | [1, 1,048,576] |

------

### 2.2 String类型配置（Type=string时必填）

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

> **关键规则**：所有数值参数的最大值默认应小于 2^20（即 1,048,576）

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

### 2.3 TokenId类型配置（Type=tokenid时必填）

```python
"TokenIdConfig" : {
    "RequestSize": int   # 单请求token数量
}
```

------

## 三. 配置示例

### 3.1 String类型示例

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
                "Var": 100,        # 标准差10
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

### 3.2 TokenId类型示例

#### 长文本压力测试

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 1000,
    "TokenIdConfig": {
        "RequestSize": 2048   # 单请求2048个token
    }
}
```

#### 短文本性能测试

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 5000,
    "TokenIdConfig": {
        "RequestSize": 128    # 短文本处理场景
    }
}
```

------

## 四. 常见问题

### Q1: 选择分布类型？

- **均匀分布**：适用于压力测试的基线场景
- **高斯分布**：模拟真实场景中的平均请求长度
- **Zipf分布**：生成长尾分布数据（如1%的请求占50%的计算量）
- **分布组合建议**：
   - **压力测试**：Input使用zipf分布，Output使用uniform分布
   - **稳定性测试**：Input/Output均使用gaussian分布

------

## 五. 注意事项

1. **Token类型必填项**：使用tokenid类型时，必须在模型配置文件中的`path`参数路径下传递有效的、包含`tokenizer_config.json`（词汇表）文件的模型权重路径

2. **特殊值处理**：当MinValue=MaxValue时生成固定长度序列