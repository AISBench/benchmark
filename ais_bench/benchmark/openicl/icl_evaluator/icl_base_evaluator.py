"""Base Evaluator."""

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from datasets import Dataset
from scipy.stats import hypergeom

from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import ICLE_CODES
from ais_bench.benchmark.utils.logging.exceptions import PredictionInvalidException, AISBenchImplementationError


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k.
    
    Args:
        n (int): Total number of samples.
        c (int): Number of correct samples.
        k (int): Top k samples.

    Returns:
        float: pass@k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _compute_g_pass_at_k(n: int, c: int, k: int, m: int) -> float:
    """Compute g pass@k.
    
    Args:
        n (int): Total number of samples.
        c (int): Number of correct samples.
        k (int): Top k samples.
        m (int): Top m samples.
    """
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    return hypergeom.sf(m - 1, n, c, k)


def compute_g_pass_at_k(n: int, c: int, k: int, t: float) -> float:
    """Compute g pass@k.
    
    Args:
        n (int): Total number of samples.
        c (int): Number of correct samples.
        k (int): Top k samples.
        t (float): Top t samples.
    """
    m = max(int(np.ceil(k * t)), 1)
    return _compute_g_pass_at_k(n, c, k, m)

class BaseEvaluator:

    def __init__(self) -> None:
        self._dataset_replica_idx = 0  # Default value for dataset_replica_idx
        self.logger = AISLogger()


    @property
    def dataset_replica_idx(self):
        return self._dataset_replica_idx


    def group(self, n: int, details: List[Dict[str, Any]],
              test_set: Dataset) -> Dict[str, Any]:
        """Group the details by the example abbreviation.
        
        Args:
            n (int): Number of replicas.
            details (List[Dict[str, Any]]): Details of the evaluation.
            test_set (Dataset): Test dataset.

        Returns:
            Dict[str, Any]: Dictionary of example abbreviations and their replications.
        """
        example2replications = {}
        resample_details = []
        step = len(details) // n
        for i in range(step):
            for detail in details[i::step]:
                resample_details.append(detail)
        for detail, example in zip(details, test_set):
            example_abbr = f"{example['subdivision']}_{example['idx']}"
            if example_abbr not in example2replications:
                example2replications[example_abbr] = []
            example.update({'detail': detail})
            example2replications[example_abbr].append(example)
        for _, replications in example2replications.items():
                if len(replications) != n:
                    raise PredictionInvalidException(
                        ICLE_CODES.REPLICATION_LENGTH_MISMATCH,
                        message=f"Replication length mismatch: {len(replications)} != {n}",
                    )

        return example2replications

    def reduce(self, details: List[Dict[str, Any]], k_list: List[int], n_val: int) -> Dict[str, Any]:
        """Aggregate results.
        
        Args:
            details (List[Dict[str, Any]]): Details of the evaluation.
            k_list (List[int]): List of top k samples.
            n_val (int): Number of replicas.

        Returns:
            dict: Aggregated results.
        """
        eval_results = OrderedDict()

        # Step 1: Global Sample Accuracy
        # Calculate global sample accuracy - using avg@n format
        sample_accuracy = np.mean([detail[f'avg@{n_val}'] for detail in details])
        eval_results[f'avg@{n_val}'] = 100 * sample_accuracy
        
        # For each k value, compute global pass@k and cons@k
        for k_val in k_list:
            # Global pass@k
            pass_at_k = np.mean([detail[f'pass@{k_val}'] for detail in details])
            eval_results[f'pass@{k_val}'] = 100 * pass_at_k
            
            # Global cons@k (majority voting accuracy)
            cons_at_k = np.mean([detail[f'cons@{k_val}'] for detail in details])
            eval_results[f'cons@{k_val}'] = 100 * cons_at_k

        # Step 2: Category-wise Breakdown Statistics
        subdivision_map = defaultdict(list)
        for detail in details:
            try:
                # Extract category names (first part of example_abbr)
                subdiv = detail['example_abbr'].split('_')[0]
                subdivision_map[subdiv].append(detail)
            except KeyError:
                # Ignore records without example_abbr
                continue
        
        # Process by category
        if len(subdivision_map) > 1:
            for subdiv, sub_details in sorted(subdivision_map.items()):
                # Category-level avg@n
                sub_avg = np.mean([d.get(f'avg@{n_val}', 0.0) for d in sub_details])
                eval_results[f'{subdiv}/avg@{n_val}'] = 100 * sub_avg
                
                # Category-level pass@k & cons@k
                for k_val in k_list:
                    sub_pass = np.mean([d.get(f'pass@{k_val}', 0.0) for d in sub_details])
                    eval_results[f'{subdiv}/pass@{k_val}'] = 100 * sub_pass
                    
                    sub_cons = np.mean([d.get(f'cons@{k_val}', 0.0) for d in sub_details])
                    eval_results[f'{subdiv}/cons@{k_val}'] = 100 * sub_cons
        
        # Step 3: Preserve Raw Data
        # Define core metric names to exclude (avoid duplication)
        CORE_METRICS = {
            'example_abbr', 'predictions',  # Metadata fields
            # Dynamic metrics
            f'avg@{n_val}', 
            *{f'pass@{k}' for k in k_list},
            *{f'cons@{k}' for k in k_list},
            # Metrics with category prefixes
            *{f'{subdiv}/avg@{n_val}' for subdiv in subdivision_map},
            *{f'{subdiv}/pass@{k}' for subdiv in subdivision_map for k in k_list},
            *{f'{subdiv}/cons@{k}' for subdiv in subdivision_map for k in k_list}
        }
        
        # Dynamically extract all non-core fields
        all_fields = set()
        for detail in details:
            all_fields |= detail.keys()
        extra_fields = all_fields - CORE_METRICS
        
        # Non-core field processing
        for field in sorted(extra_fields):
            field_values = [d[field] for d in details if field in d]
            if not field_values:  # Skip empty fields
                continue
                
            try:
                # Numeric fields - calculate global average
                global_mean = np.mean(field_values)
                eval_results[field] = 100 * global_mean
                
                # Category-level aggregation
                for subdiv in subdivision_map:
                    sub_vals = [d[field] for d in subdivision_map[subdiv] if field in d]
                    if sub_vals:
                        try:
                            eval_results[f'{subdiv}/{field}'] = 100 * np.mean(sub_vals)
                        except TypeError:
                            eval_results[f'{subdiv}/{field}'] = sub_vals
            except (TypeError, ValueError):
                # Non-numeric types - preserve original values
                eval_results[field] = field_values
                
                # Category-level preservation (no aggregation)
                for subdiv in subdivision_map:
                    sub_vals = [d[field] for d in subdivision_map[subdiv] if field in d]
                    if sub_vals:
                        eval_results[f'{subdiv}/{field}'] = sub_vals

        return eval_results

    def pred_postprocess(self, predictions: List) -> Dict:
        if not hasattr(self, 'pred_postprocessor') or self.pred_postprocessor is None:
            return predictions
        else:
            kwargs = deepcopy(self.pred_postprocessor)
            proc = TEXT_POSTPROCESSORS.get(kwargs.pop('type'))
            return [proc(pred, **kwargs) for pred in predictions]


    def evaluate(
        self,
        k: Union[int, List[int]],
        n: int,
        original_dataset: Dataset,
        **score_kwargs,
    ):
        """Evaluate the predictions and references.
        
        Args:
            k (Union[int, List[int]]): Top k samples.
            n (int): Number of replicas.
            original_dataset (Dataset): Original dataset.
            **score_kwargs: Score kwargs.

        Returns:
            dict: Evaluation results.
        """
        # Check if predictions and references have the
        # same length if both are provided
        if ('predictions' in score_kwargs and 'references' in score_kwargs
                and score_kwargs['references'] is not None):
            len_predictions, len_references = len(score_kwargs['predictions']), len(score_kwargs['references'])
            if len_predictions != len_references:
                raise PredictionInvalidException(
                        ICLE_CODES.PREDICTION_LENGTH_MISMATCH,
                        message=f'Predictions and references must have the same length, '
                        f'but got prediction({len_predictions}) and references({len_references})',
                    )

        real_size = len(original_dataset) // n  # dataset size of each replica
        all_details = []
        all_results = []
        k_list = [k] if isinstance(k, int) else k

        def select_fn(i: int, real_size: int, n: int, x: Any) -> Any:
            """Select the element from the i-th duplication within each group (choose one per group).
            
            Args:
                i (int): Index of the element.
                real_size (int): Number of elements in each group.
                n (int): Number of replicas.
                x (Any): Element to select.
            Returns:
                Any: Selected element.
            """
            if isinstance(x, Dataset):
                # Computing non-consecutive indices: select the i-th element in each group
                indices = [j * n + i for j in range(real_size)]
                return x.select(indices)
            elif isinstance(x, Iterable):
                return [x[j * n + i] for j in range(real_size)]
            else:
                return x

        # Run evaluation for each replica
        for i in range(n):
            self._dataset_replica_idx = i
            self.logger.info(f'Running {i+1}-th replica of evaluation')

            current_params = {
                key: select_fn(i, real_size, n, value)
                for key, value in score_kwargs.items()
            }
            current_params['predictions'] = self.pred_postprocess(
                current_params['predictions'])
            results = self.score(**current_params)
            details = results.pop('details', None)
            if details is not None:
                if isinstance(details, Dict):
                    details = list(details.values())
                all_details.extend(details)
            all_results.append(results)

        eval_results = {}
        for single_replica_results in all_results:
            for key in single_replica_results:
                if key not in eval_results:
                    eval_results[key] = []
                eval_results[key].append(single_replica_results[key])
        for key in deepcopy(eval_results):
            if isinstance(eval_results[key][0], float) or isinstance(
                    eval_results[key][0], int):
                if n > 1:
                    new_key = "accuracy" if key == "pass@1" else key
                    eval_results[new_key + f' ({n} runs average)'] = np.mean(
                        eval_results[key])
                    eval_results.pop(key)
                else:
                    eval_results[key] = np.mean(eval_results[key])

        # Calculate the additional metrics
        grouped_examples = self.group(n, all_details, original_dataset)
        can_calculate = False

        if len(all_details) != 0:
            eval_details = []

            for example_abbr, examples in grouped_examples.items():
                detail = {'predictions': [], 'example_abbr': example_abbr}
                c = 0

                for example in examples:
                    detail['predictions'].append(example['detail'])
                    
                    correct_key = None
                    for key in ['correct', 'is_correct', 'cascade_correct']:
                        if key in example['detail']:
                            correct_key = key
                            break
                    
                    if correct_key:
                        can_calculate = True
                        c += int(example['detail'][correct_key])

                if can_calculate and n > 1 and max(k_list) > 1:
                    total_samples = len(examples)

                    # avg@n = Number of correct samples / Total number of samples
                    detail[f'avg@{n}'] = c / total_samples

                    for _k in k_list:
                        detail[f'pass@{_k}'] = compute_pass_at_k(n=total_samples, c=c, k=_k)
                        if c > _k / 2:
                            detail[f'cons@{_k}'] = 1.0
                        else:
                            detail[f'cons@{_k}'] = 0.0

                eval_details.append(detail)

            if can_calculate and n > 1 and max(k_list) > 1:
                eval_results.update(self.reduce(eval_details, k_list, n))

            # Store eval_details in eval_results
            eval_results['details'] = eval_details

            # Process details to flatten the predictions
            for detail in eval_details:
                # Extract all prediction fields and flatten them
                flattened_predictions = {}
                for pred in detail['predictions']:
                    for k, v in pred.items():
                        if k not in flattened_predictions:
                            flattened_predictions[k] = [v]
                        else:
                            flattened_predictions[k].append(v)

                # Replace the predictions list with the flattened dictionary
                for k, v in flattened_predictions.items():
                    detail[k] = v

                # Remove the original predictions field
                detail.pop('predictions')
            
            return eval_results

        # If there are no details, return results
        return results
    

    def score(self):
        raise AISBenchImplementationError(ICLE_CODES.UNKNOWN_ERROR, 
                                           f"Method {self.__class__.__name__} hasn't been implemented yet")

    @staticmethod
    def is_num_equal(predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        else:
            return
