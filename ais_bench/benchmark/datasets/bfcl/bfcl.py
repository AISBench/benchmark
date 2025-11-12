import json
from typing import Dict
import uuid
import os.path as osp
from os import environ
from datasets import Dataset

from .bfcl_dependency import *
from ..base import BaseDataset

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS, LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging import get_logger


# Version prefix for BFCL dataset files
VERSION_PREFIX = "BFCL_v3"


def encode_fields(data):
    """
    Encode non-string fields in data to JSON strings for serialization.
    
    Args:
        data: List of dictionaries containing dataset items
        
    Returns:
        List of dictionaries with non-string fields encoded as JSON strings
    """
    # Fields that need to be encoded if they are not already strings
    fields = [
        "question",
        "ground_truth",
        "function",
        "missed_function",
        "involved_classes",
        "initial_config",
    ]
    for item in data:
        for field in fields:
            if field in item and not isinstance(item[field], str):
                item[field] = json.dumps(item[field], ensure_ascii=False)
    return data


@LOAD_DATASET.register_module()
class BFCLDataset(BaseDataset):
    """
    BFCL Dataset Loader
    
    Handles loading and preprocessing of BFCL benchmark datasets for different
    categories and programming languages.
    """

    @staticmethod
    def load(path: str, category: str, test_ids: list[str] = None):
        """
        Load BFCL dataset from specified path and category.
        
        Args:
            path: Path to the dataset directory (uses default if None)
            category: Dataset category (e.g., 'python', 'java', 'javascript', 'relevance')
            test_ids: Optional list of specific test IDs to load
            
        Returns:
            Dataset: HuggingFace Dataset object containing the loaded data
            
        Raises:
            ImportError: If required BFCL dependencies are not installed
            ValueError: If dataset and ground truth have mismatched lengths or IDs
        """
        logger = get_logger()
        
        # Check if BFCL dependencies are installed
        if not BFCL_INSTALLED:
            raise ImportError(
                "Missing required package 'bfcl-eval'. To run BFCL evaluation, "
                "install it via: pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps"
            )

        # Use default prompt path if not specified
        if not path:
            real_path = PROMPT_PATH
        else:
            real_path = path
            
        # Construct dataset file path
        dataset_path = get_data_path(
            osp.join(real_path, f"{VERSION_PREFIX}_{category}.json")
        )
        gt_path = osp.join(
            real_path, f"possible_answer/{VERSION_PREFIX}_{category}.json"
        )

        # Determine if this is a relevance category (special handling)
        is_relevance_category = "relevance" in category

        # Create a copy of test_ids for tracking used IDs
        test_case_ids_to_generate = test_ids[:] if test_ids else []

        # Load main dataset file
        dataset = []
        with open(dataset_path, "r") as f:
            for line in f:
                line = json.loads(line)
                dataset.append(line)

        # Load or generate ground truth data
        ground_truth = []
        if not is_relevance_category:
            # Load ground truth from file for non-relevance categories
            gt_path = get_data_path(
                osp.join(real_path, f"possible_answer/{VERSION_PREFIX}_{category}.json")
            )
            with open(gt_path, "r") as f:
                for line in f:
                    line = json.loads(line)
                    ground_truth.append(line)
        else:
            # Generate mock ground truth for relevance categories
            ground_truth = [
                {"id": d["id"], "ground_truth": ["relevance_mock_gt"]} for d in dataset
            ]
            
        # Validate that dataset and ground truth have matching lengths
        if len(dataset) != len(ground_truth):
            raise ValueError(
                f"Dataset and ground truth have different lengths: {len(dataset)} != {len(ground_truth)}"
            )
            
        # Process and filter data based on test_ids
        data = []
        used_ids = []
        for d, gt in zip(dataset, ground_truth):
            if test_ids:
                if d["id"] not in test_ids:
                    continue
                used_ids.append(
                    test_case_ids_to_generate.pop(
                        test_case_ids_to_generate.index(d["id"])
                    )
                )
            # Validate that dataset and ground truth IDs match
            if d["id"] == gt["id"]:
                d["ground_truth"] = gt["ground_truth"]
                data.append(d)
            else:
                raise ValueError(
                    f"Dataset and ground truth have different ids: {d['id']} != {gt['id']}"
                )
                
        # Process multi-turn test cases and encode fields
        data = process_multi_turn_test_case(data)
        data = encode_fields(data)
        
        # Log warnings for missing test IDs and info about used IDs
        if test_ids:
            if len(test_case_ids_to_generate) != 0:
                logger.warning(
                    f"Test ids not all exist in the dataset: {test_case_ids_to_generate}"
                )
            logger.info(f"Experimenting on test ids of {category}: {used_ids}")
            
        return Dataset.from_list(data)


class BFCLEvaluator(BaseEvaluator):
    """
    Base class for BFCL evaluators.
    
    Provides common functionality for evaluating function calling capabilities
    across different programming languages and evaluation scenarios.
    """
    
    def __init__(self, category: str, is_fc_model=True):
        """
        Initialize BFCL evaluator.
        
        Args:
            category: Dataset category (determines programming language)
            is_fc_model: Whether the model supports function calling format
        """
        super().__init__()
        self.is_fc_model = is_fc_model
        self.category = category
        # Generate unique model name for tracking
        self.model_name = "function-call-model-" + str(uuid.uuid4()).split("-")[-1]
        
        # Determine programming language based on category
        self.language = "Python"
        if is_java(category):
            self.language = "Java"
        if is_js(category):
            self.language = "JavaScript"

    def score(self, predictions, references):
        """
        Abstract method for scoring predictions against references.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Raises:
            NotImplementedError: This is an abstract class
        """
        raise NotImplementedError(
            "BFCLEvaluator is an abstract class and should not be used directly."
        )

    def decode_ast(self, result, language="Python"):
        """
        Decode Abstract Syntax Tree (AST) from model result.
        
        Args:
            result: Model output to decode
            language: Programming language for AST decoding
            
        Returns:
            List of decoded function calls
        """
        if self.is_fc_model:
            # For function calling models, parse JSON and extract function calls
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            # For non-function calling models, use default prompting decoder
            return default_decode_ast_prompting(result, language)


@ICL_EVALUATORS.register_module()
class BFCLRelevanceEvaluator(BFCLEvaluator):
    """
    Evaluator for BFCL relevance tests.
    
    Tests whether models correctly identify when function calls are relevant
    or irrelevant to the given context.
    """
    
    def score(self, predictions, references, test_set):
        """
        Score relevance predictions.
        
        Args:
            predictions: Model predictions for relevance tests
            references: Ground truth references
            test_set: Test dataset containing questions and metadata
            
        Returns:
            Dictionary containing accuracy metrics and detailed results
        """
        details = []
        correct_count = 0
        
        for i in range(len(predictions)):
            index: str = test_set[i]["id"]
            
            # Parse model result based on model type
            if self.is_fc_model:
                try:
                    model_result_item = json.loads(predictions[i])
                except json.JSONDecodeError:
                    model_result_item = predictions[i]
            else:
                model_result_item = predictions[i]
                
            contain_func_call = False
            decoded_result = None
            decode_error = None

            # Try to decode the model output to check for function calls
            try:
                model_result_item_raw = model_result_item
                decoded_result = self.decode_ast(
                    model_result_item_raw, language="Python"
                )
                # Decode successfully, which means the model output is in valid function call format
                contain_func_call = True
                if is_empty_output(decoded_result):
                    # Empty output is not considered as a valid function call
                    contain_func_call = False
            except Exception as e:
                # Decode failed, which means the model output is not in valid function call format
                contain_func_call = False
                decode_error = str(e)

            # Determine success based on test type
            # Irrelevance test means no function call should be outputted
            if "irrelevance" in index:
                success = not contain_func_call
            else:
                success = contain_func_call

            if success:
                correct_count += 1
            else:
                # Record error details for failed cases
                temp = {}
                temp["id"] = index
                temp["prompt"] = test_set[i]["question"]
                temp["origin_prediction"] = model_result_item
                temp["predictions"] = decoded_result
                temp["correct"] = success
                if "irrelevance" in index:
                    temp["error"] = [
                        f"Valid syntax. Successfully decode AST when it should not."
                    ]
                    temp["error_type"] = "irrelevance_error:decoder_success"
                else:
                    temp["error"] = [
                        f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"
                    ]
                    temp["error_type"] = "relevance_error:decoder_failed"
                details.append(temp)

        score = correct_count / len(predictions)

        return {
            "accuracy": score,
            "correct_count": correct_count,
            "total_count": len(predictions),
            "details": details,
        }


@ICL_EVALUATORS.register_module()
class BFCLMultiTurnEvaluator(BFCLEvaluator):
    """
    Evaluator for BFCL multi-turn conversation tests.
    
    Tests function calling capabilities in multi-turn conversational contexts
    where the model needs to maintain context across multiple interactions.
    """

    def decode_execute(self, result, is_fc_model=True):
        """
        Decode execution result from model output.
        
        Args:
            result: Model output to decode
            is_fc_model: Whether the model supports function calling format
            
        Returns:
            List of executable function calls
        """
        if is_fc_model:
            if isinstance(result, str):
                result = json.loads(result)
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    def score(self, predictions, references, test_set):
        """
        Score multi-turn predictions.
        
        Args:
            predictions: Model predictions for multi-turn tests
            references: Ground truth references
            test_set: Test dataset containing questions and metadata
            
        Returns:
            Dictionary containing accuracy metrics and detailed results
            
        Raises:
            AssertionError: If predictions and references have different lengths
        """
        # Validate that predictions and references have matching lengths
        assert len(predictions) == len(
            references
        ), f"The length of the model result ({len(predictions)}) does not match the length of the prompt ({len(references)}). Please check the input files for completeness."

        details = []
        correct_count = 0
        
        for i in range(len(predictions)):
            index: str = test_set[i]["id"]
            
            # Model result is stored as a list of list of model responses. Each inner list represents a turn.
            try:
                multi_turn_model_result_list: list[list] = json.loads(predictions[i])
            except json.JSONDecodeError:
                multi_turn_model_result_list: list[list] = []
                
            multi_turn_ground_truth_list: list[list[str]] = json.loads(references[i])
            test_entry: dict = test_set[i]
            
            # Parse JSON fields in test entry
            test_entry["initial_config"] = json.loads(test_entry["initial_config"])
            test_entry["involved_classes"] = json.loads(test_entry["involved_classes"])

            # Remove the function doc from the score file for better readability; they are repeated and way too long
            if "function" in test_entry:
                del test_entry["function"]

            # Check if model output is in correct format (list of lists)
            if type(multi_turn_model_result_list) != list:
                details.append(
                    {
                        "id": index,
                        "correct": False,
                        "error": {
                            "error_message": [
                                "Error during inference phase. Model did not output a list of model responses."
                            ],
                            "error_type": "multi_turn:inference_error",
                        },
                        "prompt": test_entry["question"],
                        "model_result": multi_turn_model_result_list,
                        "possible_answer": multi_turn_ground_truth_list,
                    }
                )
                
            # Check if force-terminated during inference phase.
            # This happens when the model has retried too many times and still haven't figured out the answer.
            # When force-terminated, no further evaluation is needed. This whole entry will be failed.
            if len(multi_turn_model_result_list) != len(multi_turn_ground_truth_list):
                details.append(
                    {
                        "id": index,
                        "correct": False,
                        "error": {
                            "error_message": [
                                f"Model was force-terminated during inference phase. The length of the model result turns ({len(multi_turn_model_result_list)}) does not match the length of the ground truth turns ({len(multi_turn_ground_truth_list)})."
                            ],
                            "error_type": "multi_turn:force_terminated",
                        },
                        "prompt": test_entry["question"],
                        "model_result": multi_turn_model_result_list,
                        "possible_answer": multi_turn_ground_truth_list,
                    }
                )
                continue

            # Decode model results into executable function calls
            multi_turn_model_result_list_decoded: list[list[list[str]]] = (
                []
            )  # decode_execute returns a list of strings
            
            # Try decoding the model results into executable function calls
            for single_turn_model_result_list in multi_turn_model_result_list:
                single_turn_model_result_list_decoded = []
                for model_result_item in single_turn_model_result_list:
                    # model_result_item is per step
                    try:
                        decoded_result: list[str] = self.decode_execute(
                            model_result_item, self.is_fc_model
                        )
                        if is_empty_execute_response(decoded_result):
                            # Empty output is not considered as a valid function call
                            continue

                        single_turn_model_result_list_decoded.append(decoded_result)

                    except Exception as e:
                        # Ignore any failed decoding and continue to the next message
                        # We only care about the decoded function call, not the error message or if the model is chatting
                        continue
                        
                multi_turn_model_result_list_decoded.append(
                    single_turn_model_result_list_decoded
                )

            # Check if the model output the correct function calls
            accuracy_checker_result = multi_turn_checker(
                multi_turn_model_result_list_decoded,
                multi_turn_ground_truth_list,
                test_entry,
                self.category,
                self.model_name,
            )

            if not accuracy_checker_result["valid"]:
                # Record error details for failed cases
                temp = {}
                temp["id"] = index
                temp["prompt"] = test_entry["question"]
                temp["origin_prediction"] = multi_turn_model_result_list
                temp["predictions"] = multi_turn_model_result_list_decoded
                temp["references"] = multi_turn_ground_truth_list
                temp["correct"] = accuracy_checker_result.pop("valid", False)
                temp["error"] = make_json_serializable(accuracy_checker_result)
                details.append(temp)
            else:
                correct_count += 1

        score = correct_count / len(predictions)

        return {
            "accuracy": score,
            "correct_count": correct_count,
            "total_count": len(predictions),
            "details": details,
        }


@ICL_EVALUATORS.register_module()
class BFCLSingleTurnEvaluator(BFCLEvaluator):
    """
    Evaluator for BFCL single-turn function calling tests.
    
    Tests function calling capabilities in single-turn contexts where the model
    needs to generate appropriate function calls based on the given prompt.
    """
    
    def score(self, predictions, references, test_set):
        """
        Score single-turn predictions.
        
        Args:
            predictions: Model predictions for single-turn tests
            references: Ground truth references
            test_set: Test dataset containing questions and metadata
            
        Returns:
            Dictionary containing accuracy metrics and detailed results
        """
        details = []
        correct_count = 0
        
        for i in range(len(predictions)):
            index: str = test_set[i]["id"]
            
            # Parse model result based on model type
            if self.is_fc_model:
                try:
                    model_result_item = json.loads(predictions[i])
                except json.JSONDecodeError:
                    model_result_item = predictions[i]
            else:
                model_result_item = predictions[i]
                
            # Parse function prompt and possible answer
            prompt_item = json.loads(test_set[i]["function"])
            possible_answer_item = json.loads(references[i])

            # Try to decode AST from model result
            try:
                model_result_item_raw = model_result_item
                model_result_item = self.decode_ast(
                    model_result_item, language="Python"
                )
            except Exception as e:
                # Record AST decoding failure
                details.append(
                    {
                        "id": index,
                        "correct": False,
                        "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
                        "error_type": "ast_decoder:decoder_failed",
                        "prompt": test_set[i]["question"],
                        "model_result_raw": model_result_item_raw,
                        "possible_answer": possible_answer_item,
                    }
                )
                continue

            # Check if decoded output is in correct function calling format
            decoder_output_valid = is_function_calling_format_output(model_result_item)
            if not decoder_output_valid:
                details.append(
                    {
                        "id": index,
                        "correct": False,
                        "error": [
                            "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                        ],
                        "error_type": "ast_decoder:decoder_wrong_output_format",
                        "prompt": test_set[i]["question"],
                        "model_result_raw": str(model_result_item_raw),
                        "model_result_decoded": str(model_result_item),
                        "possible_answer": possible_answer_item,
                    }
                )
                continue

            # Check accuracy using AST checker
            checker_result = ast_checker(
                prompt_item,
                model_result_item,
                possible_answer_item,
                self.language,
                self.category,
                "qwen3-32b-FC",
            )

            if checker_result["valid"]:
                correct_count += 1
            else:
                # Record error details for failed cases
                temp = {}
                temp["id"] = index
                temp["prompt"] = test_set[i]["question"]
                temp["origin_prediction"] = model_result_item_raw
                temp["predictions"] = model_result_item
                temp["references"] = possible_answer_item
                temp["correct"] = checker_result["valid"]
                temp["error"] = checker_result["error"]
                temp["error_type"] = checker_result["error_type"]
                details.append(temp)

        score = correct_count / len(predictions)
        return {
            "accuracy": score,
            "correct_count": correct_count,
            "total_count": len(predictions),
            "details": details,
        }
