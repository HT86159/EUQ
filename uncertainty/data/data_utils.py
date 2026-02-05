"""Data Loading Utilities."""
import os
import pandas as pd
from datasets import Dataset
from uncertainty.utils import utils
parser = utils.get_parser()
args, unknown = parser.parse_known_args()

def transform_answers(answer):
    return {'text': [answer], 'answer_start': [0]}

def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""
    user = os.environ['USER']
    train_dataset, validation_dataset = None, None
    if dataset_name == "hallucination":
        benchmark_dir = os.path.join(args.image_path, 'hallucination.csv')
        df = pd.read_csv(benchmark_dir)
        result = []
        for _, row in df.iterrows():
            question_parts = [str(row['question']), '']
            for option in ['A', 'B', 'C', 'D']:
                question_parts.append(f"{option}. {str(row[option])}")
            question_str = " ".join(filter(None, question_parts))
            question_str += "(only answer a full option, do not need explanation)"
            features = {
                'id': row['image'],
                'title': row['hallucination_type'],
                'context': None,
                'question': question_str,
                'answers': {"text":[f"{row['answer']} {row[row['answer']]}"], 'answer_start': [0]}
            }
            result.append(features)

        df_new = pd.DataFrame(result)
        validation_dataset = Dataset.from_pandas(df_new)
        features = validation_dataset.features
        train_dataset = Dataset.from_pandas(df_new)

    else:
        raise ValueError

    return train_dataset, validation_dataset
