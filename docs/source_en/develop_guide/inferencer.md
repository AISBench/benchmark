# Inferencer Overview

Inferencer is the core component in AISBench responsible for executing model inference. It connects datasets, retrievers (Retriever), and models, and is responsible for sending processed prompts to models for inference and collecting and managing inference results.

## Core Functions

Inferencer undertakes the following core responsibilities in AISBench's evaluation workflow:

1. **Data Preparation**: Get data list from retriever (Retriever), including input prompts, ground truth, and other information
2. **Model Invocation**: Adopt different methods to call models for inference according to model type (API models or local models)
   - **API Models**: Call service inference interfaces through async HTTP requests
   - **Local Models**: Directly call locally loaded models for batch inference
3. **Result Management**: Collect, process, and save inference results, including:
   - Model-generated text content
   - Inference status (success/failure)
   - Performance metrics (such as latency, throughput, etc., in performance mode)
   - Error information (if inference fails)
4. **Status Tracking**: In performance evaluation mode, track and statistics request status, including:
   - Number of sent requests (post)
   - Number of received responses (rev)
   - Number of failed requests (failed)
   - Number of completed requests (finish)

## Architecture Design

Inferencer adopts a layered design, including the following base classes:

- **BaseInferencer**: Base class for all inferencers, providing common functions such as model building and output processing
- **BaseApiInferencer**: Base class for API model inferencers, providing async request processing, status tracking, and other functions
- **BaseLocalInferencer**: Base class for local model inferencers, providing batch inference, data loading, and other functions

Inferencers can inherit from both `BaseApiInferencer` and `BaseLocalInferencer` as needed to support both API models and local models.

## Currently Supported Inferencer Types

AISBench currently supports the following inferencer types:

### 1. GenInferencer (Generative Inferencer)

**Function**: Inferencer for generative tasks, supporting text generation, question answering, and other tasks.

**Features**:

- Supports both API models and local models
- Supports streaming and non-streaming inference
- Supports performance evaluation mode
- Supports custom stopping criteria (stopping_criteria)

**Use Cases**:

- Text generation tasks
- Question answering tasks
- Code generation tasks
- Mathematical reasoning tasks

**Implementation File**: [icl_gen_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

### 2. MultiTurnGenInferencer (Multi-turn Dialogue Inferencer)

**Function**: Inferencer for multi-turn dialogue tasks, supporting multi-turn interactive dialogue scenarios.

**Features**:

- Supports both API models and local models
- Supports multiple inference modes:
  - `every`: Round-by-round inference, using model's previous round output as next round input
  - `last`: Only infer the last round
  - `every_with_gt`: Round-by-round inference, but using ground truth instead of model output
- Supports performance evaluation mode

**Use Cases**:

- Multi-turn dialogue tasks
- Tasks requiring contextual interaction
- Conversational question answering tasks

**Implementation File**: [icl_multiturn_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_multiturn_inferencer.py)

### 3. PPLInferencer (Perplexity Inferencer)

**Function**: Inferencer for Perplexity evaluation, selecting answers by calculating the perplexity of each option, mainly used for multiple choice question (MCQ) tasks.

**Features**:

- Only supports API models (does not support local models)
- Does not support streaming inference
- Does not support performance evaluation mode
- Selects the option with the lowest perplexity as the prediction result by calculating the perplexity of each candidate answer

**Use Cases**:

- Multiple choice question (MCQ) tasks
- Classification tasks requiring selection based on perplexity

**Implementation File**: [ppl_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/ppl_inferencer.py)

### 4. BFCLV3FunctionCallInferencer (Function Call Inferencer)

**Function**: Inferencer for function call tasks, supporting scenarios where models call external functions or tools.

**Features**:

- Only supports API models
- Supports multi-turn function calls
- Supports holdout function mechanism
- Supports result processing and feedback for function calls

**Use Cases**:

- Function call tasks
- Tool usage tasks
- Tasks requiring models to call external APIs

**Implementation File**: [icl_bfcl_v3_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_bfcl_v3_inferencer.py)

## Inferencer Selection Guide

Select the appropriate inferencer according to different task types and model types:

| Task Type | Model Type | Recommended Inferencer |
|---------|---------|-----------|
| Text generation, Question answering | API models | GenInferencer |
| Text generation, Question answering | Local models | GenInferencer |
| Multi-turn dialogue | API models | MultiTurnGenInferencer |
| Multi-turn dialogue | Local models | MultiTurnGenInferencer |
| Multiple choice questions (MCQ) | API models | PPLInferencer |
| Function calls | API models | BFCLV3FunctionCallInferencer |

## Relationship with Related Components

Inferencer closely collaborates with other components in AISBench's evaluation workflow:

1. **Relationship with Retriever**:
   - Inferencer gets data from Retriever through the `get_data_list` method
   - Retriever is responsible for generating in-context examples and prompts

2. **Relationship with Model**:
   - Inferencer calls the Model's `generate` method for inference
   - For API models, inferencer calls model services through HTTP requests
   - For local models, inferencer directly calls model instances

3. **Relationship with OutputHandler**:
   - Inferencer uses OutputHandler to manage and save inference results
   - Different types of inferencers use different OutputHandlers

4. **Relationship with Dataset**:
   - Inferencer gets inference-related parameters from Dataset configuration
   - Parameters such as `max_out_len` can be obtained from dataset configuration

## Further Reading

- [Supporting New Inferencers](./new_inferencer.md): Learn how to implement custom inferencers
- [Prompt Template](../prompt/prompt_template.md): Learn about prompt template definitions
- [Meta Template](../prompt/meta_template.md): Learn about model meta template definitions
- [Retriever](../prompt/retriever.md): Learn about how retrievers work

