from collections import OrderedDict
import transformers

DBFS_PATH = "dbfs:/FileStore/tables/summarization"

class Summarizer:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

    def __call__(self, texts):
        self._init_model()
        return [
            self._apply_model(text)
            for text in texts
        ]

    def _init_model(self):
        self.model = transformers.pipeline(
            "summarization",
            model=self.model_name_or_path,
            framework="pt"
        )

    def _apply_model(self, text):
        return self.model(text, truncation=True)[0]["summary_text"]


class SplitSummarizer(Summarizer):
    def __init__(self, model_name_or_path, n_splits, min_sents_per_split=5):
        super().__init__(model_name_or_path)
        self.n_splits = n_splits
        self.min_sents_per_split = min_sents_per_split

    def _apply_model(self, text):
        lines = text.splitlines()
        lines_per_split = max(self.min_sents_per_split, len(lines) // self.n_splits)
        res = []
        for start_idx in range(0, len(lines), lines_per_split):
            part = "\n".join(lines[start_idx:start_idx + lines_per_split])
            res.append(super()._apply_model(part))

        return "\n".join(res)


def _infer_model_path(model_name_or_path, temp_s3_bucket):
    if not model_name_or_path.startswith("dbfs:"):
        return model_name_or_path

    s3_path = model_name_or_path.replace(DBFS_PATH, temp_s3_bucket)
    assert s3_path != model_name_or_path, f"Models should be stored in subdirs under {DBFS_PATH}"
    return s3_path

def _infer_model_path_and_download(model_name_or_path, temp_s3_bucket):
    s3_path = _infer_model_path(model_name_or_path, temp_s3_bucket)
    if not s3_path.startswith("s3:/"):
        return model_name_or_path

    import fsspec
    import os
    local_file = "/databricks/driver/models/" + s3_path.split("/")[-1]
    if not os.path.exists(local_file):
        fs, _, _ = fsspec.get_fs_token_paths(s3_path)
        fs.get(s3_path, local_file, recursive=True)
    return local_file

def run_model_on_partition(partition, temp_s3_bucket):
    model_names = set()
    uids = []
    texts = []
    for obj in partition:
        model_names.add(obj[0])
        uids.append(obj[1][0])
        texts.append(obj[1][1])
    assert len(model_names) == 1, f"Expecting only one model per partition, found: {model_names}"

    model_name = list(model_names)[0]
    model = models[model_name]
    model.model_name_or_path = _infer_model_path_and_download(model.model_name_or_path, temp_s3_bucket)
    print(f"Running model '{model_name}' on {len(texts)} texts")
    preds = model(texts)
    return [(model_name, dict(zip(uids, preds)))]


models = OrderedDict([
    ("BART xsum-samsum", Summarizer("Salesforce/bart-large-xsum-samsum")),
    ("BART dialogsum", Summarizer(f"{DBFS_PATH}/models/bart-dialogsum")),
    ("BART dialogsum 2 parts", SplitSummarizer(f"{DBFS_PATH}/models/bart-dialogsum", 2)),
    ("BART wiley-100", Summarizer("s3://usw2-sfdc-ecp-prod-databricks-users/databricks_2051_ai-prod-00Dd0000000eekuEAA/20220403_20220503/models/bart_dialogsum-finetuned-wiley-100-issue-resolution/checkpoint-54")),
    ("BART wiley-full", Summarizer("s3://usw2-sfdc-ecp-prod-databricks-users/databricks_2051_ai-prod-00Dd0000000eekuEAA/20220403_20220503/models/bart_dialogsum-finetuned-wiley-18525-issue-resolution/checkpoint-5785")),
    ("BART wiley-erez", Summarizer("s3://usw2-sfdc-ecp-prod-databricks-users/databricks_2051_ai-prod-00Dd0000000eekuEAA/20220403_20220503/models/bart_dialogsum-finetuned-wiley-18525-issue-resolution-erez/checkpoint-6942")),
    ("BART cnn/dm", Summarizer("facebook/bart-large-cnn")),
])
