# 🧭 AISBench插件开发指南（简略初版）
## 🔬 AISBench插件原理
![xx](../img/plugin/plugin_uml.png)
- AISBench工具所有的运行流程中的各种关键环节的功能通过对应一种类承载，每种类通过多态的来实现某个环节的不同功能。
- 实现同一个环节的不同类都通过同一个注册器来注册（例如图中的`Object MODEL`就是注册所有模型运行类的注册器），在运行时根据注册器的注册信息来动态选择使用哪个类。所有的注册器请参考代码[registry.py](../ais_bench/benchmark/registry.py)。
- 在aisbench的原生代码中，可被注册的类都写在[ais_bench/benchmark](../ais_bench/benchmark)目录内的子目录下，例如这段代码说明可被注册的模型运行类需要写在[ais_bench/benchmark/models](../ais_bench/benchmark/models)路径下：
    ```python
    MODELS = Registry('model', locations=get_locations('models'))
    ```
- AISBench的插件本质上是拓展可被注册的类的所在路径，使得写在在插件中的某个路径的类(例如模型运行类)也可被AISBench中的注册器(例如`Object MODELS`)注册使用。AISBench是通过**python的entry_point机制**获取到插件包的路径，将此路径加入可被注册的`location`中。

## AISBench插件快速实现
可参考本`README.md`所在的`plugin_examples`目录下的代码，代码结构介绍：
```shell
├─ais_bench_plugin_example_pkg # 插件包名称，可自定义，与entry_point中的路径相同
│  ├─clients # 插件包中拓展的client类
│  │  ├─__init__.py
│  │  └─example_client.py
│  └─models # 插件包中拓展的model类
│     ├─__init__.py
│     └─example_model.py
└─config_example # AISBench自定义配置文件
   └─perf_example.py
```
### setup.py中配置entry_point
其中`setup.py`需要配置entry_point相关内容：
```python
# ......
setup(
    name='ais_bench_plugin_example',
    version='0.0.1',
    description='ais_bench_plugin_example',
    long_description=long_description,
    packages=packages,
    include_package_data=True,
    keywords='ais_bench_plugin_example',
    python_requires='>=3.8.0',
    entry_points={
        'ais_bench.benchmark_plugins': [ # 不可修改，固定为'ais_bench.benchmark_plugins'
            'example_plugin = ais_bench_plugin_example_pkg', # <插件entry_point> = <导包路径，以`.`分隔>
        ],
    },
)
```
其中`ais_bench_plugin_example_pkg`对应`ais_bench_plugin_example_pkg/`文件所在的包路径。`ais_bench_plugin_example_pkg/`文件夹下的子文件夹名称需要与[registry.py](../ais_bench/benchmark/registry.py)中各注册器初始化时用的子目录名称相同（例如`MODELS = Registry('model', locations=get_locations('models'))`中`get_locations`的入参）

### 实现插件包中自己想定义的类
> ⚠️注：本样例中的client类`ExampleModel`的实现完全拷贝自`ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat`，本样例中的model类`ExampleClient`的实现完全拷贝自`ais_bench.benchmark.clients.openai_chat_text_client.OpenAIChatTextClient`。

拓展的自定义的类需要加上对应种类的装饰器，例如自定义一个client类：
```python
from abc import ABC
from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.registry import CLIENTS

@CLIENTS.register_module() # CLIENTS注册器的register_module方法作为装饰器
class ExampleClient(BaseClient, ABC):
    # ......
```

可拓展的类的信息如下表格（待完善）：
| 类类型 | 类描述 | 承载类的子路径名 |导入注册器|
| --- | --- | -------- | --------- |
| 模型运行类 | 模型运行类的实现 | `models` | `from ais_bench.benchmark.registry import MODELS`|
| 客户端类 | 客户端类的实现 | `clients` | `from ais_bench.benchmark.registry import CLIENTS` |

### 实现自定义的启动配置文件
参考[config_example/perf_example.py](../config_example/perf_example.py):

```python
from mmengine.config import read_base
from ais_bench_plugin_example_pkg.models import ExampleModel # 导入样例中自定义的模型运行类
from ais_bench_plugin_example_pkg.clients import ExampleClient # 导入样例中自定义的请求客户端类
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen import synthetic_datasets

datasets = [
    *synthetic_datasets,
]

models = [
    dict(
        attr="service",
        type=ExampleModel, # 使用自定义的模型运行类
        abbr='example-model',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        trust_remote_code=False,
        custom_client=dict(type=ExampleClient), # 使用自定义的请求客户端类
        generation_kwargs = dict(
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]


infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalRunner,
                 max_num_workers=2,
                 task=dict(type=OpenICLApiInferTask)), )

work_dir = 'outputs/example_model/'
```

## AISBench插件安装使用（以样例为例）
1. 安装AISBench工具
参考[AISBench工具安装](https://gitee.com/aisbench/benchmark#-%E5%B7%A5%E5%85%B7%E5%AE%89%E8%A3%85)
2. 安装插件包：
```shell
# 在当前README.md所在目录
pip install -e .
```
3. 按实际修改`config_example/perf_example.py`中的配置
4. 通过自定义配置文件启动性能评测
```shell
ais_bench config_example/perf_example.py --mode perf --debug
```


