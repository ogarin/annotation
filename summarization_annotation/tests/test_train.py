import datasets
import transformers


# This is a test for debugging BART training


pipeline = transformers.pipeline("summarization", "Salesforce/bart-large-xsum-samsum")
metric = datasets.load_metric("rouge")

def preprocess_function(examples):
    inputs = [doc for doc in examples["dialogue"]]
    tokenizer = pipeline.tokenizer
    model_inputs = tokenizer(inputs, truncation=True, return_tensors="pt")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"].tolist(), truncation=True, return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

batch_size = 4
args = transformers.Seq2SeqTrainingArguments(
    "bart-wiley-20k",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    report_to="none",
#     fp16=True,
)

data_collator = transformers.DataCollatorForSeq2Seq(pipeline.tokenizer, model=pipeline.model)

import nltk
import numpy as np


def compute_metrics(eval_pred):
    tokenizer = pipeline.tokenizer
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def fake_collator(features, return_tensors=None):
    print(features[0])
    raise NotImplemented()


import pandas as pd

dataset = datasets.load_dataset("samsum")


split_dfs = {
    "train": pd.DataFrame(dataset['train'][:10]),
    "dev": pd.DataFrame(dataset['validation'][:10]),
}

trainer = transformers.Seq2SeqTrainer(
    pipeline.model,
    args,
    train_dataset=preprocess_function(split_dfs["train"].head(10)),
    eval_dataset=preprocess_function(split_dfs["dev"].head(10)),
    data_collator=data_collator,
    #     data_collator=fake_collator,
    tokenizer=pipeline.tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
