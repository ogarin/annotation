import logging
import os
from ast import literal_eval
from types import SimpleNamespace
from typing import List

from robustnessgym import Dataset, Spacy, CachedOperation
from robustnessgym.core.constants import CACHEDOPS
from robustnessgym.core.tools import strings_as_json
from robustnessgym.logging.utils import set_logging_level
from spacy import load
from spacy.attrs import DEP, IS_ALPHA, IS_PUNCT, IS_STOP, LEMMA, LOWER, TAG, SENT_END, \
    SENT_START, ORTH, POS, ENT_IOB
from spacy.tokens import Doc

from align import BertscoreAligner, NGramAligner, StaticEmbeddingAligner

set_logging_level('critical')
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def _spacy_encode(self, x):
    arr = x.to_array(
        [DEP, IS_ALPHA, IS_PUNCT, IS_STOP, LEMMA, LOWER, TAG, SENT_END, SENT_START,
         ORTH, POS, ENT_IOB])
    return {
        'arr': arr.flatten(),
        'shape': list(arr.shape),
        'words': [t.text for t in x]
    }


def _spacy_decode(self, x):
    doc = Doc(self.nlp.vocab, words=x['words'])
    return doc.from_array(
        [DEP, IS_ALPHA, IS_PUNCT, IS_STOP, LEMMA, LOWER,
         TAG, SENT_END, SENT_START, ORTH, POS, ENT_IOB],
        x['arr'].reshape(x['shape'])
    )


Spacy.encode = _spacy_encode
Spacy.decode = _spacy_decode


class AlignerCap(CachedOperation):
    def __init__(
            self,
            aligner,
            spacy,
            **kwargs,
    ):
        super(AlignerCap, self).__init__(**kwargs)
        self.spacy = spacy
        self.aligner = aligner

    @classmethod
    def encode(cls, x):
        # Convert to built-in types from np.int / np.float
        return super(AlignerCap, cls).encode([
            {str(k): [(int(t[0]), float(t[1])) for t in v] for k, v in d.items()}
            for d in x
        ])

    @classmethod
    def decode(cls, x):
        x = super(AlignerCap, cls).decode(x)
        x = [{literal_eval(k): v for k, v in d.items()} for d in x]
        return x

    def apply(self, batch, columns, *args, **kwargs):
        # Run the aligner on the first example of the batch
        return [
            self.aligner.align(
                self.spacy.retrieve(batch, columns[0])[0],
                [self.spacy.retrieve(batch, col)[0] for col in columns[1:]]
                if len(columns) > 2 else
                [self.spacy.retrieve(batch, columns[1])[0]],
            )
        ]


class BertscoreAlignerCap(AlignerCap):
    def __init__(
            self,
            threshold: float,
            top_k: int,
            spacy,
    ):
        super(BertscoreAlignerCap, self).__init__(
            aligner=BertscoreAligner(threshold=threshold, top_k=top_k),
            spacy=spacy,
            threshold=threshold,
            top_k=top_k,
        )


class NGramAlignerCap(AlignerCap):
    def __init__(
            self,
            spacy,
    ):
        super(NGramAlignerCap, self).__init__(
            aligner=NGramAligner(),
            spacy=spacy
        )


class StaticEmbeddingAlignerCap(AlignerCap):
    def __init__(
            self,
            threshold: float,
            top_k: int,
            spacy,
    ):
        super(StaticEmbeddingAlignerCap, self).__init__(
            aligner=StaticEmbeddingAligner(threshold=threshold, top_k=top_k),
            spacy=spacy,
            threshold=threshold,
            top_k=top_k,
        )


def _run_aligners(
        dataset: Dataset,
        aligners: List[CachedOperation],
        doc_column: str,
        reference_column: str,
        summary_columns: List[str] = None,
):
    if not summary_columns:
        summary_columns = []

    to_columns = []
    if reference_column is not None:
        to_columns.append(reference_column)
    to_columns.extend(summary_columns)

    for aligner in aligners:

        # Run the aligner on (document, summary) pairs

        dataset = aligner(
            dataset,
            [doc_column] + to_columns,
            # Must use `batch_size = 1`
            batch_size=1,
        )

        if reference_column is not None and len(summary_columns):
            # Run the aligner on (reference, summary) pairs
            dataset = aligner(
                dataset,
                [reference_column] + summary_columns,
                # Must use `batch_size = 1`
                batch_size=1,
            )

        if len(to_columns) > 1:
            # Instead of having one column for (document, summary) comparisons, split
            # off into (1 + |summary_columns|) total columns, one for each comparison

            # Retrieve the (document, summary) column
            doc_summary_column = aligner.retrieve(
                dataset[:],
                [doc_column] + to_columns,
            )[tuple([doc_column] + to_columns)]

            for i, col in enumerate(to_columns):
                # Add as a new column after encoding with the aligner's `encode` method
                dataset.add_column(
                    column=str(aligner.identifier(columns=[doc_column, col])),
                    values=[aligner.encode([row[i]]) for row in doc_summary_column],
                )

            # Remove the (document, summary) column
            dataset.remove_column(
                str(
                    aligner.identifier(
                        columns=[doc_column] + to_columns
                    )
                )
            )
            del dataset.interactions[CACHEDOPS].history[
                (
                    aligner.identifier,
                    strings_as_json(
                        strings=[doc_column] + to_columns
                    )
                )
            ]

        if reference_column is not None and len(summary_columns) > 1:
            # Instead of having one column for (reference, summary) comparisons, split
            # off into (|summary_columns|) total columns, one for each comparison

            # Retrieve the (reference, summary) column
            reference_summary_column = aligner.retrieve(
                dataset[:],
                [reference_column] + summary_columns,
            )[tuple([reference_column] + summary_columns)]

            for i, col in enumerate(summary_columns):
                # Add as a new column
                dataset.add_column(
                    column=str(aligner.identifier(columns=[reference_column, col])),
                    values=[
                        aligner.encode([row[i]]) for row in reference_summary_column
                    ]
                )

            # Remove the (reference, summary) column
            dataset.remove_column(
                str(
                    aligner.identifier(
                        columns=[reference_column] + summary_columns
                    )
                )
            )
            del dataset.interactions[CACHEDOPS].history[
                (
                    aligner.identifier,
                    strings_as_json(
                        strings=[reference_column] + summary_columns
                    )
                )
            ]

    return dataset


def deanonymize_dataset(
        rg_path: str,
        standardized_dataset: Dataset,
        processed_dataset_path: str = None,
        n_samples: int = None,

):
    """Take an anonymized dataset and add back the original dataset columns."""
    assert processed_dataset_path is not None, \
        "Please specify a path to save the dataset."

    # Load the dataset
    dataset = Dataset.load_from_disk(rg_path)

    if n_samples:
        dataset.set_visible_rows(list(range(n_samples)))
        standardized_dataset.set_visible_rows(list(range(n_samples)))

    text_columns = []

    # Add columns from the standardized dataset
    dataset.add_column('document', standardized_dataset['document'])
    text_columns.append('document')

    if 'summary:reference' in standardized_dataset.column_names:
        dataset.add_column('summary:reference', standardized_dataset['summary:reference'])
        text_columns.append('summary:reference')

    # Preprocessing all the text columns
    dataset = dataset.update(
        lambda x:
            {
                f'preprocessed_{k}': x[k] if args.no_clean else clean_text(x[k])
                for k in text_columns
            }
    )

    # Run the Spacy pipeline on all preprocessed text columns
    nlp = load('en_core_web_lg')
    nlp.add_pipe('sentencizer', before="parser")
    spacy = Spacy(nlp=nlp)
    dataset = spacy(
        dataset,
        [f'preprocessed_{col}' for col in text_columns],
        batch_size=100,
    )

    # Directly save to disk
    dataset.save_to_disk(processed_dataset_path)

    return dataset


def run_workflow(
        jsonl_path: str = None,
        dataset: Dataset = None,
        doc_column: str = None,
        reference_column: str = None,
        summary_columns: List[str] = None,
        bert_aligner_threshold: float = 0.5,
        bert_aligner_top_k: int = 3,
        embedding_aligner_threshold: float = 0.5,
        embedding_aligner_top_k: int = 3,
        processed_dataset_path: str = None,
        n_samples: int = None,
        anonymize: bool = False,
):
    assert (jsonl_path is None) != (dataset is None), \
        "One of `jsonl_path` and `dataset` must be specified."
    assert processed_dataset_path is not None, \
        "Please specify a path to save the dataset."

    # Load the dataset
    if jsonl_path is not None:
        dataset = Dataset.from_jsonl(jsonl_path)

    if doc_column is None:
        # Assume `doc_column` is called "document"
        doc_column = 'document'
        assert doc_column in dataset.column_names, \
            f"`doc_column={doc_column}` is not a column in dataset."
        print("Assuming `doc_column` is called 'document'.")

    if reference_column is None:
        # Assume `reference_column` is called "summary:reference"
        reference_column = 'summary:reference'
        print("Assuming `reference_column` is called 'summary:reference'.")
        if reference_column not in dataset.column_names:
            print("No reference summary loaded")
            reference_column = None

    if summary_columns is None or len(summary_columns) == 0:
        # Assume `summary_columns` are prefixed by "summary:"
        summary_columns = []
        for col in dataset.column_names:
            if col.startswith("summary:") and col != "summary:reference":
                summary_columns.append(col)
        print(f"Reading summary columns from dataset. Found {summary_columns}.")

    if len(summary_columns) == 0 and reference_column is None:
        raise ValueError("At least one summary is required")

    # Set visible rows to restrict to the first `n_samples`
    if n_samples:
        dataset.set_visible_rows(list(range(n_samples)))

    # Combine the text columns into one list
    text_columns = [doc_column] + ([reference_column] if reference_column else []) + summary_columns

    # Preprocessing all the text columns
    dataset = dataset.update(
        lambda x: {
            f'preprocessed_{k}': x[k]
            for k in text_columns
        }
    )

    # Run the Spacy pipeline on all preprocessed text columns
    nlp = load('en_core_web_lg')
    nlp.add_pipe('sentencizer', before="parser")
    spacy = Spacy(nlp=nlp)
    dataset = spacy(
        dataset,
        [f'preprocessed_{col}' for col in text_columns],
        batch_size=100,
    )

    # Run the 3 align pipelines
    bert_aligner = BertscoreAlignerCap(
        threshold=bert_aligner_threshold,
        top_k=bert_aligner_top_k,
        spacy=spacy,
    )

    embedding_aligner = StaticEmbeddingAlignerCap(
        threshold=embedding_aligner_threshold,
        top_k=embedding_aligner_top_k,
        spacy=spacy,
    )

    ngram_aligner = NGramAlignerCap(
        spacy=spacy,
    )

    dataset = _run_aligners(
        dataset=dataset,
        aligners=[bert_aligner, embedding_aligner, ngram_aligner],
        doc_column=f'preprocessed_{doc_column}',
        reference_column=f'preprocessed_{reference_column}' if reference_column else None,
        summary_columns=[f'preprocessed_{col}' for col in summary_columns],
    )

    # Save the dataset
    if anonymize:
        # Remove certain columns to anonymize and save to disk
        for col in [doc_column, reference_column]:
            if col is not None:
                dataset.remove_column(col)
                dataset.remove_column(f'preprocessed_{col}')
                dataset.remove_column(
                    str(spacy.identifier(columns=[f'preprocessed_{col}']))
                )
                del dataset.interactions[CACHEDOPS].history[
                    (spacy.identifier, f'preprocessed_{col}')
                ]
        dataset.save_to_disk(f'{processed_dataset_path}.anonymized')
    else:
        # Directly save to disk
        dataset.save_to_disk(processed_dataset_path)

    return dataset


def parse_prediction_jsonl_name(prediction_jsonl: str):
    """Parse the name of the prediction_jsonl to extract useful information."""
    # Analyze the name of the prediction_jsonl
    filename = prediction_jsonl.split("/")[-1]

    # Check that the filename ends with `.results.anonymized`
    if filename.endswith(".results.anonymized"):
        # Fmt: <model>-<training dataset>.<eval dataset>.<eval split>.results.anonymized

        # Split using a period
        model_train_dataset, eval_dataset, eval_split = filename.split(".")[:-2]
        model, train_dataset = model_train_dataset.split("-")

        return SimpleNamespace(
            model_train_dataset=model_train_dataset,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_split=eval_split,
        )

    raise NotImplementedError(
        "Prediction files must be named "
        "<model>-<training dataset>.<eval dataset>.<eval split>.results.anonymized. "
        f"Please rename the prediction file {filename} and run again."
    )


def join_predictions(
        dataset_jsonl: str = None,
        prediction_jsonls: str = None,
        save_jsonl_path: str = None,
):
    """Join predictions with a dataset."""
    assert prediction_jsonls is not None, "Must have prediction jsonl files."

    print(
        "> Warning: please inspect the prediction .jsonl file to make sure that "
        "predictions are aligned with the examples in the dataset. "
        "Use `get_dataset` to inspect the dataset."
    )

    # Load the dataset
    dataset = get_dataset(dataset_jsonl=dataset_jsonl)

    # Parse names of all prediction files to get metadata
    metadata = [
        parse_prediction_jsonl_name(prediction_jsonl)
        for prediction_jsonl in prediction_jsonls
    ]

    # Load the predictions
    predictions = [
        Dataset.from_jsonl(json_path=prediction_jsonl)
        for prediction_jsonl in prediction_jsonls
    ]

    # Predictions for a model
    for i, prediction_data in enumerate(predictions):
        # Get metadata for i_th prediction file
        metadata_i = metadata[i]

        # Construct a prefix for columns added to the dataset for this prediction file
        prefix = metadata_i.model_train_dataset

        # Add the predictions column to the dataset
        for col in prediction_data.column_names:
            # Don't add the indexing information since the dataset has it already
            if col not in {'index', 'ix', 'id'}:
                # `add_column` will automatically ensure that column lengths match
                if col == 'decoded':  # rename decoded to summary
                    dataset.add_column(f'summary:{prefix}', prediction_data[col])
                else:
                    dataset.add_column(f'{prefix}:{col}', prediction_data[col])

    # Save the dataset back to disk
    if save_jsonl_path:
        dataset.to_jsonl(save_jsonl_path)
    else:
        print("Dataset with predictions was not saved since `save_jsonl_path` "
              "was not specified.")

    return dataset


def standardize_dataset(
        dataset_name: str = None,
        dataset_version: str = None,
        dataset_split: str = 'test',
        dataset_jsonl: str = None,
        doc_column: str = None,
        reference_column: str = None,
        save_jsonl_path: str = None,
        no_save: bool = False,
):
    """Load a dataset from Huggingface and dump it to disk."""
    # Load the dataset from Huggingface
    dataset = get_dataset(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_split=dataset_split,
        dataset_jsonl=dataset_jsonl,
    )

    if doc_column is None:
        if reference_column is not None:
            raise ValueError("You must specify `doc_column` if you specify `reference_column`")
        try:
            doc_column, reference_column = {
                'cnn_dailymail': ('article', 'highlights'),
                'xsum': ('document', 'summary'),
                'samsum': ('dialogue', 'summary'),
            }[dataset_name]
        except:
            raise NotImplementedError(
                "Please specify `doc_column`."
            )

    # Rename the columns
    if doc_column != 'document':
        dataset.add_column('document', dataset[doc_column])
        dataset.remove_column(doc_column)
    dataset.add_column('summary:reference', dataset[reference_column])
    dataset.remove_column(reference_column)

    # Save the dataset back to disk
    if save_jsonl_path:
        dataset.to_jsonl(save_jsonl_path)

    elif (save_jsonl_path is None) and not no_save:
        # Auto-create a path to save the standardized dataset
        os.makedirs('preprocessing', exist_ok=True)
        if not dataset_jsonl:
            dataset.to_jsonl(
                f'preprocessing/'
                f'standardized_{dataset_name}_{dataset_version}_{dataset_split}.jsonl'
            )
        else:
            dataset.to_jsonl(
                f'preprocessing/'
                f'standardized_{dataset_jsonl.split("/")[-1]}'
            )

    return dataset


def get_dataset(
        dataset_name: str = None,
        dataset_version: str = None,
        dataset_split: str = 'test',
        dataset_jsonl: str = None,
):
    """Load a dataset."""
    assert (dataset_name is not None) != (dataset_jsonl is not None), \
        "Specify one of `dataset_name` or `dataset_jsonl`."

    # Load the dataset
    if dataset_name is not None:
        return get_hf_dataset(dataset_name, dataset_version, dataset_split)

    return Dataset.from_jsonl(json_path=dataset_jsonl)


def get_hf_dataset(name: str, version: str = None, split: str = 'test'):
    """Get dataset from Huggingface."""
    if version:
        return Dataset.load_dataset(name, version, split=split)
    return Dataset.load_dataset(name, split=split)

