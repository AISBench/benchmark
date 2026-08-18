from herding import generate_coreset
from herding.eval_datasets import get_eval_dataset
from herding.eval_datasets.dataset_base import require_env

CORESET_RATIO = float(require_env('CORESET_RATIO'))


def main():
    ds = get_eval_dataset()
    indices = ds.load_indices()
    ds.save_data_by_indices(indices, 'origin')
    coreset_size = round(len(indices) * CORESET_RATIO)

    modified_indices = generate_coreset(coreset_size)
    output_path = ds.save_data_by_indices(modified_indices, 'coreset')
    print(f'output: {output_path}')


if __name__ == "__main__":
    main()
