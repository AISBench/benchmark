"""
# 涉及到数值类型配置参数的最大值无特殊说明时，均应小于 2**20 （= 1 M）
# 
# StringConfig中的随机生成方法参数说明:
# -------------------------------------------------
# 包含 输入/出 分布配置 "Method" 和 输入/出 长度配置 "Params"
# 格式说明：
# 输入/出 分布名称 -- "Method" : 输入分布类型
# 输入/出 长度配置 "Params" 内各项参数的名称: 说明及取值范围
#
# [Uniform均匀分布] -- "Method" : "uniform"
#   - MinValue: 最小值，范围为 [1, 2**20]
#   - MaxValue: 最大值, 范围为 [1, 2**20], 可等于MinValue
#
# [Gaussian高斯分布] -- "Method" : "gaussian"
#   - Mean    : 平均值, 范围为 [-3.0e38, 3.0e38]，分布中心位置
#   - Var     : 方差, 范围为[0, 3.0e38]，控制数据分散程度
#   - MinValue: 最小值, 范围为 [1, 2**20], 可低于Mean
#   - MaxValue: 最大值, 范围为 [1, 2**20], 可高于Mean, 可等于MinValue
#
# [Zipf齐夫分布] -- "Method" : "zipf"
#   - Alpha   : 形状参数, 范围为(1.0,10.0], 值越大分布越均匀
#   - MinValue: 最小值, 范围为 [1, 2**20]
#   - MaxValue: 最大值, 范围为 [1, 2**20], 需大于MinValue
"""
synthetic_config = {
    "Type":"tokenid",   # [tokenid/string]，生成的随机数据集类型，支持固定长度的随机tokenid，和随机长度的string，两种类型的数据集
    "RequestCount": 10, # 生成的请求条数，应与模型侧配置文件中的 decode_batch_size 一致
    "TrustRemoteCode": False, #是否信任远端代码，tokenid模式下需要加载tokenizer生成tokenid，默认为Fasle
    "StringConfig" : {  # string类型的随机数据集的配置相关项，请参考以上注释处："StringConfig中的随机生成方法参数说明"
        "Input" : {     # 每条请求的输入长度
            "Method": "uniform",
            "Params": {"MinValue": 1, "MaxValue": 200}
        },
        "Output" : {    # 每条请求的输出长度
            "Method": "gaussian",
            "Params": {"Mean": 100, "Var": 200, "MinValue": 1, "MaxValue": 100}
        }
    },
    "TokenIdConfig" : { # tokenid类型的随机数据集的配置相关项
        "RequestSize": 10 # 每条请求的长度，即每条请求中token id的个数，应与模型侧配置文件中的 input_seq_len 一致
    }
}