import argparse


DEFAULT_CORESET_RATIO = 0.2
DEFAULT_OUTPUT_DIR = "./datasets"


def _coreset_ratio(value: str) -> float:
    """Validate coreset ratio passed from the command line."""
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a floating-point number") from exc

    if not 0 < ratio <= 1:
        raise argparse.ArgumentTypeError("must be in the range (0, 1]")
    return ratio


def parse_args(argv=None):
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Keep help and argument errors independent of the selection workflow.
    from herding.selection import run_selection

    output_path = run_selection(
        dataset_name=args.eval_dataset,
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        coreset_ratio=args.coreset_ratio,
        output_dir=args.output_dir,
    )
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
