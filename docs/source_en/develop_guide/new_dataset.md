# Supporting New Datasets and Accuracy Evaluators

Currently, AISBench supports the following data types: open-source datasets, custom datasets, and synthetic datasets. Before adapting a new dataset, it is recommended to first refer to the usage instructions for [custom datasets](../advanced_tutorials/custom_dataset.md) and [synthetic datasets](../advanced_tutorials/synthetic_dataset.md) to confirm whether they can meet actual needs.

For datasets that cannot meet requirements (for example, dataset loading methods or accuracy calculation rules differ significantly from other datasets), adaptation is needed. Before starting, it is recommended to first refer to the definition methods of [prompt_template](../prompt/prompt_template.md) and [meta_template](../prompt/meta_template.md) to understand how AISBench constructs prompts, how to convert raw data into actual model inputs, and the functions of components involved.

Specific implementation reference is as follows:

1. Add a new dataset script `mydataset.py` in the `ais_bench/benchmark/datasets` folder. This script needs to include:

   - **Dataset and its loading method**: Need to define a `MyDataset` class and implement the dataset loading method `load`. This method is a static method and needs to return data of type `datasets.Dataset`. Here we use HuggingFace Dataset as a unified interface for datasets to avoid introducing additional logic. Reference format is as follows:

    ```python
    import datasets
    from .base import BaseDataset

    class MyDataset(BaseDataset):

        @staticmethod
        def load(**kwargs) -> datasets.Dataset:
            ... # Implement dataset loading logic
            data_list = ... # Dataset list
            return datasets.Dataset.from_list(data_list)  # Convert dataset list to HuggingFace Dataset object
    ```

    It is recommended to add the new dataset class to [`__init__.py`](../../../ais_bench/benchmark/datasets/__init__.py) for convenient automatic import later.

    For specific examples, refer to [Aime2024Dataset](../../../ais_bench/benchmark/datasets/aime2024.py)

    For **multimodal data**, it is necessary to use formatted concatenation in the `load` function to concatenate text, image, video, and audio data into one data item. In subsequent parsing, data will be restored and concatenated into model input according to markers for each data type.

    Concatenation format example:

    ```text
    <AIS_TEXT_START>{text}<AIS_CONTENT_TAG><AIS_IMAGE_START>{image}<AIS_CONTENT_TAG><AIS_VIDEO_START>{video}<AIS_CONTENT_TAG><AIS_AUDIO_START>{audio}<AIS_CONTENT_TAG>
    ```

    Where `{text}`, `{image}`, `{video}`, `{audio}` are the text, image, video, and audio content in the dataset.

    For specific examples, refer to [MMCustomDataset](../../../ais_bench/benchmark/datasets/mm_custom.py)

   - **(Optional) Custom accuracy evaluator**: If the existing accuracy evaluators in AISBench cannot meet needs, users need to define a `MyDatasetEvaluator` class and implement the scoring method `score`. This method needs to return a dictionary containing metrics and their corresponding scores based on the input `predictions` and `references` lists. Since a dataset may have multiple metrics, the returned dictionary should include all relevant evaluation metrics. Specific example is as follows:

   ```python
   from typing import List
   from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator

   class MyDatasetEvaluator(BaseEvaluator):

       def score(self, predictions: List, references: List) -> dict:
           # Implement evaluation logic
           # Return format: {"metric_name": score_value, ...}
           pass
   ```

    For specific implementation, refer to [MATHEvaluator](../../../ais_bench/benchmark/datasets/math.py)

   - **(Optional) Custom post-processing method**: If the existing post-processing methods in AISBench cannot meet needs, users need to define a `mydataset_postprocess` method to get corresponding post-processed results based on the input string. This method is usually used for scenarios such as cleaning model output and extracting answers. Specific example is as follows:

   ```python
   def mydataset_postprocess(text: str) -> str:
       # Implement post-processing logic, such as extracting answers, cleaning format, etc.
       # Return processed string
       pass
   ```

2. After defining dataset loading, evaluation, and data post-processing methods, add the following configuration `my_dataset.py` in the configuration directory [../ais_bench/benchmark/configs/datasets](../../../ais_bench/benchmark/configs/datasets/my_dataset):

   ```python
   from ais_bench.benchmark.datasets import MyDataset, MyDatasetEvaluator, mydataset_postprocess

   # Accuracy evaluation configuration
   mydataset_eval_cfg = dict(
       evaluator=dict(type=MyDatasetEvaluator),  # Custom accuracy evaluator class name
       pred_postprocessor=dict(type=mydataset_postprocess)  # Custom data post-processing method
   )

   # Dataset reading configuration: Configure according to fields of each sample in the dataset, used to fill prompt_template
   mydataset_reader_cfg = dict(
       input_columns=["question"],  # Input field list
       output_column="answer"       # Output field (ground truth)
   )

   # Inference configuration
   mydataset_infer_cfg = dict(
       prompt_template=dict(
           # Prompt template class name, configure according to data type:
           # - PromptTemplate: Pure text input
           # - MultiTurnPromptTemplate: Multi-turn dialogue input
           # - MMPromptTemplate: Multimodal input
           type=PromptTemplate,
           template=dict(
               round=[
                   dict(
                       role="HUMAN",
                       prompt="{question}\nRemember to put your final answer within \\boxed{}.",
                   ),
               ],
           ),
       ),
       retriever=dict(type=ZeroRetriever),      # Retriever configuration
       inferencer=dict(type=GenInferencer),     # Inferencer configuration
   )

   # Dataset configuration list
   mydataset_datasets = [
       dict(
           type=MyDataset,                    # Custom dataset class name
           abbr='mydataset',                  # Unique dataset identifier
           # ... Other dataset initialization parameters ...
           reader_cfg=mydataset_reader_cfg,   # Dataset reading configuration
           infer_cfg=mydataset_infer_cfg,     # Inference configuration
           eval_cfg=mydataset_eval_cfg        # Accuracy evaluation configuration
       )
   ]
   ```

    Then execute the command to start local evaluation task:

    ```bash
    ais_bench --models vllm_api_stream_chat --datasets my_dataset
    ```

3. Add README documentation

   Create a `README.md` file in the configuration directory `ais_bench/benchmark/configs/datasets/my_dataset/` to explain the dataset deployment and usage methods. The README should include the following content:

   - **Dataset Introduction**: Briefly introduce the basic information, characteristics, and uses of the dataset, and attach a link to the dataset homepage (if it exists). Example format:

     ```markdown
     # MyDataset
     中文 | [English](README_en.md)
     ## Dataset Introduction
     MyDataset is a benchmark dataset for evaluating model performance on XXX tasks. The dataset contains XXX samples covering XXX different categories.

     > 🔗 Dataset homepage link [https://example.com/mydataset](https://example.com/mydataset)
     ```

   - **Dataset Deployment**: Detailed steps for downloading and deploying the dataset, including:
     - Download link or method for obtaining the dataset
     - Deployment path and directory structure requirements
     - Deployment steps (recommended to provide executable command examples)
     - Directory structure verification method (recommended to use `tree` command to show expected directory structure)

     Example format:

     ```markdown
     ## Dataset Deployment
     - The dataset package can be downloaded from the link provided by XXX 🔗 [https://example.com/mydataset.zip](https://example.com/mydataset.zip).
     - It is recommended to deploy in the `{tool root path}/ais_bench/datasets` directory (default path set in dataset tasks). Taking deployment on Linux as an example, specific execution steps are as follows:
     ```bash
     # On Linux server, in tool root path
     cd ais_bench/datasets
     wget https://example.com/mydataset.zip
     unzip mydataset.zip
     rm mydataset.zip
     ```

     - Execute `tree mydataset/` in the `{tool root path}/ais_bench/datasets` directory to view the directory structure. If the directory structure is as shown below, the dataset deployment is successful.

         ```text
         mydataset
         ├── data
         │   └── ...
         └── ...
         ```

     If the dataset is integrated through dependency packages (such as Python packages), explain the installation steps and environment requirements:

     Example format:

     ```markdown
     ## Dataset Deployment
     MyDataset dataset is integrated through Python dependency packages. Data files are included in the `mydataset-eval` dependency package and can be used directly after installing dependencies.

     ### Environment Requirements
     - **mydataset-eval** dependency package (contains complete dataset)

     ### Installation Steps
     \`\`\`bash
     pip3 install mydataset-eval
     \`\`\`
     ```

   - **(Optional) Usage Examples**: If the dataset has special usage requirements or configuration methods, provide detailed usage examples, including:
     - Model configuration examples (if the dataset requires specific model types or configurations)
     - Command examples for executing evaluation
     - Result display examples

   - **Available Dataset Tasks**: List all available dataset task configurations in table format. The table should include the following columns:
     - Task Name: Identifier for dataset configuration (used for `--datasets` parameter)
     - Introduction: Brief description of the task
     - Evaluation Metrics: Evaluation metrics used (such as accuracy, score, etc.)
     - few-shot: Number of few-shot examples (such as 0-shot, 3-shot, 5-shot, etc.)
     - Prompt Format: Prompt format type (such as string format, dialogue format, etc.)
     - Corresponding Source Configuration File Path: Relative path of the configuration file

     Example format:

     ```markdown
     ## Available Dataset Tasks
     |Task Name|Introduction|Evaluation Metrics|few-shot|Prompt Format|Corresponding Source Configuration File Path|
     | --- | --- | --- | --- | --- | --- |
     |mydataset_gen_0_shot_str|MyDataset dataset generative task|accuracy|0-shot|String format|[mydataset_gen_0_shot_str.py](mydataset_gen_0_shot_str.py)|
     |mydataset_gen_5_shot_str|MyDataset dataset generative task|accuracy|5-shot|String format|[mydataset_gen_5_shot_str.py](mydataset_gen_5_shot_str.py)|
     ```

   - **(Optional) Dataset Classification**: If the dataset contains multiple subcategories or test scenarios, they can be classified and explained according to different dimensions, for example:
     - Individual test categories: List configurations for each subcategory
     - Test groups: Explain how to batch test multiple related categories
     - Precise test configuration: Explain how to specify specific test cases for precise testing

   - **(Optional) Usage Recommendations**: Provide recommendations and notes for dataset usage to help users better use the dataset.

    For specific examples, refer to:
    - [C-Eval README](../../../ais_bench/benchmark/configs/datasets/ceval/README.md)
    - [BFCL README](../../../ais_bench/benchmark/configs/datasets/BFCL/README.md)
    - [BBH README](../../../ais_bench/benchmark/configs/datasets/bbh/README.md)

