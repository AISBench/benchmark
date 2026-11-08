拓展多模态生成类评测基准
=================
本目录介绍多模态生成结果的评测方法，其中：

* :doc:`VBench 1.0 <vbench>`：评估已生成视频的质量与语义表现，支持 GPU / NPU，官方 Standard 套件包含 16 个维度。
* :doc:`VABench <vabench>`：评估已有带音轨 MP4 的音频质量、音视频一致性、同步性及内容表现；当前接入 GPU 上的 15 项指标，提供 778 题标准集和固定 78 题子集的使用说明，并支持与原生官方结果对照。

.. toctree::
   :maxdepth: 2

   gedit_bench
   oneig
   vbench
   vabench
