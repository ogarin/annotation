from collections import OrderedDict
import transformers
import os

def _download_from_dbfs(model_name_or_path):
    model_name_or_path = (
        model_name_or_path[:-1]
        if model_name_or_path.endswith("/")
        else model_name_or_path
    )
    model_short_name = model_name_or_path.split("/")[-1]
    local_path = f"/databricks/driver/{model_short_name}"
    if not os.path.exists(f"{local_path}/config.json"):
        print(f"Copying model from {model_name_or_path} to file:{local_path}")
        dbutils.fs.cp(model_name_or_path, f"file:{local_path}", recurse=True)

    return local_path


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
        local_path = (
            _download_from_dbfs(self.model_name_or_path)
            if self.model_name_or_path.startswith("dbfs:/")
            else self.model_name_or_path
        )
        self.model = transformers.pipeline(
            "summarization",
            model=local_path,
            framework="pt"
        )

    def _apply_model(self, text):
        return self.model(text, truncation=True)[0]["summary_text"]


class SplitSummarizer(Summarizer):
    def __init__(self, model_name_or_path, n_splits=2, min_sents_per_split=5):
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



def run_model_on_partition(partition):
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
    print(f"Running model '{model_name}' on {len(texts)} texts")
    preds = model(texts)
    return [(model_name, dict(zip(uids, preds)))]


models = OrderedDict([
    ("BART xsum-samsum", Summarizer("Salesforce/bart-large-xsum-samsum")),
    ("BART cnn/dm", Summarizer("facebook/bart-large-cnn")),
    # ("BART dialogsum", Summarizer("dbfs:/FileStore/tables/summarization/models/bart-dialogsum")),
])