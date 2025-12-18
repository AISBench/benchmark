# Prompt Overview

Prompts are inputs to language models (LLMs), used to guide models to generate content or calculate perplexity (PPL). The way prompts are constructed significantly affects the performance of evaluated models. In AISBench, the process of converting datasets into a series of prompts is defined by templates.

AISBench splits templates into two parts: **data-side templates** and **model-side templates**. When evaluating models, data is processed by data-side and model-side templates in sequence, eventually converted into the input format required by models.

- **Data-side templates** are called [prompt_template](./prompt_template.md), which define how to convert dataset fields into prompts.

- **Model-side templates** are called [meta_template](./meta_template.md), which define how models convert these prompts into their expected input format.

Through this separation design, AISBench achieves decoupling between dataset configuration and model configuration, allowing the same dataset to adapt to different models, and the same model to adapt to different datasets.

