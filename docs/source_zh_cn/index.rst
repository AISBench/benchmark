欢迎来到 AISBench 评测工具中文教程 ✨
=======================================

🌏 简介
-----------------------------

AISBench Benchmark 是基于 `OpenCompass <https://github.com/open-compass/opencompass>`_ 构建的模型评测工具，兼容 OpenCompass 的配置体系、数据集结构与模型后端实现，并在此基础上扩展了对服务化模型的支持能力。

当前，AISBench 支持两大类推理任务的评测场景：

🔍 精度测评：支持对服务化模型和本地模型在各类问答、推理基准数据集上的精度验证。

🚀 性能测评：支持对服务化模型的延迟与吞吐率评估，并可进行压测场景下的极限性能测试。


👉 推荐上手路径
----------------------------
为了帮助你快速上手 AISBench 评测工具，我们推荐按照以下顺序进行学习：

* 对于想要使用 AISBench 评测工具的用户，建议先阅读 :doc:`安装指南 <get_started/install>`，确保环境配置正确。
* 本教程提供的 :doc:`快速入门 <get_started/quick_start>` 将引导你完成基本的精度评测配置和运行。
* :doc:`数据集准备指南 <get_started/datasets>` 将帮助你了解支持的数据集及其准备方法。
* 基础教程部分将介绍 :doc:`评测场景介绍 <base_tutorials/scenes_intro/index>` 、:doc:`评测结果说明 <base_tutorials/results_intro/index>` 以及 :doc:`详细参数说明 <base_tutorials/all_params/index>` 等内容，帮助你更好地理解主要的评测场景的使用。
* 如果想要更深入地了解 AISBench 评测工具的高级用法，可以参考 :doc:`进阶教程 <advanced_tutorials/run_custom_config>`。
* 你可以参考 :doc:`最佳实践<best_practices/practice_nvidia>` 部分，了解在不同场景下使用 AISBench 评测工具的最佳实践。
* 最后，你可以参考 :doc:`常见问题 <faqs/faq>` 部分，解决在使用 AISBench 评测工具过程中遇到的问题。

.. toctree::
   :maxdepth: 2
   :caption: 🚀 开始你的第一步
   :hidden:

   get_started/install
   get_started/quick_start
   get_started/datasets

.. toctree::
   :maxdepth: 1
   :caption: 🧭 基础教程
   :hidden:

   base_tutorials/scenes_intro/index
   base_tutorials/results_intro/index
   base_tutorials/all_params/index

.. toctree::
   :maxdepth: 2
   :caption: 🔬 进阶教程
   :hidden:

   advanced_tutorials/run_custom_config
   advanced_tutorials/stable_stage
   advanced_tutorials/rps_distribution
   advanced_tutorials/multiturn_benchmark
   advanced_tutorials/synthetic_dataset
   advanced_tutorials/custom_dataset

.. toctree::
   :maxdepth: 2
   :caption: 💪 最佳实践
   :hidden:

   best_practices/practice_nvidia
   best_practices/practice_ascend
   best_practices/replicate_llm_datasets_accuracy

.. toctree::
   :maxdepth: 2
   :caption: ❓常见问题
   :hidden:

   faqs/faq
   faqs/error_codes

.. toctree::
   :maxdepth: 2
   :caption: 👨‍💻 开发者指南
   :hidden:

   develop_guide/contributing
   develop_guide/new_model
   develop_guide/new_dataset
   develop_guide/new_inferencer

.. toctree::
   :maxdepth: 2
   :caption: 📝 提示词工程
   :hidden:

   prompt/prompt_template
   prompt/meta_template
   prompt/overview
   prompt/retriever

.. toctree::
   :maxdepth: 2
   :caption: 🏷️ 其他
   :hidden:

   others/others