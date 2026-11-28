"""Orchestrate dataset exports and herding selection independently of the CLI."""

from pathlib import Path


CORESET_METHOD = "herding"


def _model_name(model_path: str) -> str:
    name = Path(model_path.rstrip("/\\")).name
    if not name:
        raise ValueError(f"Unable to infer model name from model path: {model_path!r}")
    return name


def run_selection(
    *,
    dataset_name: str,
    dataset_path: str,
    model_path: str,
    coreset_ratio: float,
    output_dir: str,
):
    """Export the source data and return the path to the selected coreset.

    The caller supplies a coreset ratio already validated to be in (0, 1].
    """
    from herding import generate_coreset
    from herding.eval_datasets import get_eval_dataset

    model_output_dir = (
        Path(output_dir) / dataset_name / CORESET_METHOD / _model_name(model_path)
    )
    eval_dataset = get_eval_dataset(
        dataset_name,
        dataset_path=dataset_path,
        output_dir=str(model_output_dir),
    )

    indices = eval_dataset.load_indices()
    if not indices:
        raise ValueError("The evaluation dataset is empty; cannot generate a coreset.")

    eval_dataset.save_data_by_indices(indices, "origin")
    coreset_size = max(1, round(len(indices) * coreset_ratio))
    selected_indices = generate_coreset(
        coreset_size,
        eval_dataset=eval_dataset,
        model_path=model_path,
    )
    return eval_dataset.save_data_by_indices(selected_indices, "coreset")
