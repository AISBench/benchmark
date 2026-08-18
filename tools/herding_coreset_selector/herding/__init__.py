from .features import load_model, generate_logits
from .algorithm import features_to_coreset_matrix, coreset_indices
from .eval_datasets import get_eval_dataset

from tqdm import tqdm
import time


def generate_coreset(coreset_size):
    model, tokenizer = load_model()
    eval_dataset = get_eval_dataset()

    prompts_generator = eval_dataset.dataset_prompts()
    logits_generator = generate_logits(model, tokenizer, prompts_generator)
    logits_generator = tqdm(
        logits_generator,
        total=eval_dataset.dataset_size(),
        desc="features",
    )
    logits_matrix = features_to_coreset_matrix(logits_generator)

    start = time.perf_counter()
    indices = coreset_indices(logits_matrix, coreset_size)
    print(f'    herding: {time.perf_counter() - start:.2f}s')
    return indices
