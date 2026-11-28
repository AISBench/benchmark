from datasets import Dataset


def _project_qa_test_split(dataset):
    """Replace the test split with the three fields used by QA adapters."""
    test = dataset['test']
    rows = [
        {
            'input': test['input'][i],
            'context': test['context'][i],
            'answers': test['answers'][i],
        }
        for i in range(len(test))
    ]
    dataset['test'] = Dataset.from_list(rows)
    return dataset
