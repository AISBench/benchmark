import time


def generate_coreset(coreset_size, eval_dataset, model_path):
    """Generate coreset indices for an initialized evaluation dataset."""
    # Keep package import lightweight so `python -m herding --help` does not
    # import torch/transformers/numpy before command-line arguments are parsed.
    from tqdm import tqdm

    from .algorithm import coreset_indices, features_to_coreset_matrix
    from .features import generate_logits, load_model

    model, tokenizer = load_model(model_path)

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
    print(f"    herding: {time.perf_counter() - start:.2f}s")
    return indices
