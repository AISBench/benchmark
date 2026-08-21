import argparse
from pathlib import Path


DEFAULT_CORESET_RATIO = 0.2
DEFAULT_OUTPUT_DIR = "./datasets"
CORESET_METHOD = "herding"


def _coreset_ratio(value: str) -> float:
    """Validate coreset ratio passed from the command line."""
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a floating-point number") from exc

    if not 0 < ratio <= 1:
        raise argparse.ArgumentTypeError("must be in the range (0, 1]")
    return ratio


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select representative evaluation samples with RBF Kernel Herding "
            "using hidden-state features extracted from a Hugging Face model."
        )
    )
    parser.add_argument(
        "--eval-dataset",
        required=True,

        help="Evaluation dataset adapter to use.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            "Directory containing the source evaluation dataset "
            "(for example, gpqa_diamond.csv for GPQA)."
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face model path used to extract hidden-state features.",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=_coreset_ratio,
        default=DEFAULT_CORESET_RATIO,
        help=(
            "Fraction of the original dataset selected for the coreset, in (0, 1]. "
            f"Default: {DEFAULT_CORESET_RATIO}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root directory for generated results. The final path is "
            "<output-dir>/<dataset>/herding/<model-name>/. "
            f"Default: {DEFAULT_OUTPUT_DIR}."
        ),
    )
    return parser.parse_args()


def _model_name(model_path: str) -> str:
    normalized = model_path.rstrip("/\\")
    name = Path(normalized).name
    if not name:
        raise ValueError(f"Unable to infer model name from model path: {model_path!r}")
    return name


def main():
    args = parse_args()

    # Heavy dependencies are imported only after CLI parsing, so
    # `python -m herding --help` remains a lightweight, self-contained entry.
    from herding import generate_coreset
    from herding.eval_datasets import get_eval_dataset

    output_dir = (
        Path(args.output_dir)
        / args.eval_dataset
        / CORESET_METHOD
        / _model_name(args.model_path)
    )

    eval_dataset = get_eval_dataset(
        args.eval_dataset,
        dataset_path=args.dataset_path,
        output_dir=str(output_dir),
    )

    indices = eval_dataset.load_indices()
    if not indices:
        raise ValueError("The evaluation dataset is empty; cannot generate a coreset.")

    eval_dataset.save_data_by_indices(indices, "origin")
    coreset_size = max(1, round(len(indices) * args.coreset_ratio))

    selected_indices = generate_coreset(
        coreset_size,
        eval_dataset=eval_dataset,
        model_path=args.model_path,
    )
    output_path = eval_dataset.save_data_by_indices(selected_indices, "coreset")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
