Welcome to AISBench Benchmark Tool English Tutorial ✨
=======================================

🌏 Introduction
-----------------------------

AISBench Benchmark is a model evaluation tool built on `OpenCompass <https://github.com/open-compass/opencompass>`_, compatible with OpenCompass's configuration system, dataset structure, and model backend implementation, while extending support for service-based models.

Currently, AISBench supports two major types of inference task evaluation scenarios:

🔍 Accuracy Evaluation: Supports accuracy verification of service-based models and local models on various question-answering and reasoning benchmark datasets.

🚀 Performance Evaluation: Supports the assessment of latency and throughput for service-oriented models, and enables extreme performance testing under stress test scenarios.


👉 Recommended Getting Started Path
----------------------------
To help you quickly get started with AISBench Benchmark Tool, we recommend learning in the following order:

* For users who want to use AISBench Benchmark Tool, it is recommended to first read the :doc:`Installation Guide <get_started/install>` to ensure correct environment configuration.
* The :doc:`Quick Start <get_started/quick_start>` provided in this tutorial will guide you through basic accuracy evaluation configuration and execution.
* The :doc:`Dataset Preparation Guide <base_tutorials/all_params/datasets>` will help you understand the supported datasets and how to prepare them for evaluation.
* The Basic Tutorial section will introduce :doc:`Evaluation Scenario Introduction <base_tutorials/scenes_intro/index>`, :doc:`Evaluation Result Explanation <base_tutorials/results_intro/index>`, and :doc:`Detailed Parameter Description <base_tutorials/all_params/index>` to help you better understand the use of major evaluation scenarios.
* For a deeper understanding of advanced usage of AISBench Benchmark Tool, you can refer to the :doc:`Advanced Tutorial <advanced_tutorials/run_custom_config>`.
* You can refer to the :doc:`Best Practices <best_practices/practice_nvidia>` section to learn best practices for using AISBench Benchmark Tool in different scenarios.
* Finally, you can refer to the :doc:`Frequently Asked Questions <faqs/faq>` section to solve problems encountered during the use of AISBench Benchmark Tool.

.. toctree::
   :maxdepth: 2
   :caption: 🚀 Get Started
   :hidden:

   get_started/install
   get_started/quick_start
   base_tutorials/all_params/datasets

.. toctree::
   :maxdepth: 1
   :caption: 🧭 Basic Tutorials
   :hidden:

   base_tutorials/scenes_intro/index
   base_tutorials/results_intro/index
   base_tutorials/all_params/index

.. toctree::
   :maxdepth: 2
   :caption: 🔬 Advanced Tutorials
   :hidden:

   advanced_tutorials/run_custom_config
   advanced_tutorials/stable_stage
   advanced_tutorials/rps_distribution
   advanced_tutorials/multiturn_benchmark
   advanced_tutorials/synthetic_dataset
   advanced_tutorials/custom_dataset

.. toctree::
   :maxdepth: 2
   :caption: 💪 Best Practices
   :hidden:

   best_practices/practice_nvidia
   best_practices/practice_ascend
   best_practices/replicate_llm_datasets_accuracy

.. toctree::
   :maxdepth: 2
   :caption: ❓ FAQs
   :hidden:

   faqs/faq
   faqs/error_codes

.. toctree::
   :maxdepth: 2
   :caption: 👨‍💻 Developer Guide
   :hidden:

   develop_guide/contributing
   develop_guide/new_model
   develop_guide/new_dataset
   develop_guide/new_inferencer

.. toctree::
   :maxdepth: 2
   :caption: 📝 Prompt Engineering
   :hidden:

   prompt/prompt_template
   prompt/meta_template
   prompt/overview
   prompt/retriever

.. toctree::
   :maxdepth: 2
   :caption: 🏷️ Others
   :hidden:

   others/others