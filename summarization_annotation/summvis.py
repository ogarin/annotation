import operator

import re
import streamlit as st
import spacy
from spacy.tokens import Doc
from robustnessgym import Identifier, Spacy

from align import NGramAligner, BertscoreAligner, StaticEmbeddingAligner
from components import MainView
from preprocessing import NGramAlignerCap, StaticEmbeddingAlignerCap, \
    BertscoreAlignerCap
from preprocessing import _spacy_decode, _spacy_encode

import annotation
import databricks
from worker_models import models

MIN_SEMANTIC_SIM_THRESHOLD = 0.1
MAX_SEMANTIC_SIM_TOP_K = 10

Doc.set_extension("name", default=None, force=True)
Doc.set_extension("column", default=None, force=True)
Doc.set_extension("score", default=None, force=True)

from rouge_score.rouge_scorer import RougeScorer
rouge_types = ["rouge1"]#, "rouge2", "rougeL", "rougeLsum"]
rouge_scorer = RougeScorer(rouge_types=rouge_types, use_stemmer=True)
def get_rouge_score(pred, ref):
    return rouge_scorer.score(pred, ref)['rouge1'].fmeasure

class Instance():
    def __init__(self, id_, document, reference, preds, data=None):
        self.id = id_
        self.document = document
        self.reference = reference
        self.preds = preds
        self.data = data

# @st.cache(allow_output_mutation=True)
# def get_models():
#     return OrderedDict([
#         (key, transformers.pipeline("summarization", model=name_or_path, framework="pt"))
#         for key, name_or_path in [
#             ('bart-cnn', "facebook/bart-large-cnn"),
#             ('bart-samsum', "Salesforce/bart-large-xsum-samsum"),
#         ]
#     ])

@st.cache(allow_output_mutation=True)
def get_nlp():
    return spacy.load("en_core_web_lg")

def _to_doc(text, column, name, score=None):
    if column == "document":
        # make sure spacy will treat different lines as different sentences
        text = re.sub("(?<=\w)\s*\n\s*", ".\n", text)
    document = nlp(text)
    if column != "document":
        # add newlines to summary sentences
        document = nlp("\n".join(
            sent.text
            for sent in document.sents
        ))
    document._.column = column
    document._.name = name
    document._.score = score
    return document

def retrieve(batch, index):
    if index >= len(batch["chats"]):
        st.error(f"Index {index} exceeds dataset length.")

    data = batch["chats"][index]
    uid = data["uid"]

    dialogue = data["chat_text"]
    gold_summary = data["case"].get("Description")

    document = _to_doc(dialogue, "document", "Document")
    reference = None if not gold_summary else _to_doc(gold_summary, "summary:reference", "Reference")

    preds = [
        _to_doc(
            data["preds"][model_name],
            model_name,
            model_name,
            get_rouge_score(data['preds'][model_name], gold_summary)
        )
        for model_name in models.keys()
        if model_name in data.get("preds", {})
    ]
    # for model_name, model in get_models().items():
    #     summary = model(dialogue, truncation=True)[0]["summary_text"]

    return Instance(
        id_=uid,
        document=document,
        reference=reference,
        preds=preds,
        data=data,
    )


def filter_alignment(alignment, threshold, top_k):
    filtered_alignment = {}
    for k, v in alignment.items():
        filtered_matches = [(match_idx, score) for match_idx, score in v if score >= threshold]
        if filtered_matches:
            filtered_alignment[k] = sorted(filtered_matches, key=operator.itemgetter(1), reverse=True)[:top_k]
    return filtered_alignment


def select_comparison(example):
    all_summaries = []

    if example.reference:
        all_summaries.append(example.reference)
    if example.preds:
        all_summaries.extend(example.preds)

    from_documents = [example.document]
    if example.reference:
        from_documents.append(example.reference)
    document_names = [document._.name for document in from_documents]
    if len(document_names) == 1:
        selected_document = from_documents[0]
    else:
        select_document_name = sidebar_placeholder_from.selectbox(
            label="Comparison FROM:",
            options=document_names
        )
        document_index = document_names.index(select_document_name)
        selected_document = from_documents[document_index]

    remaining_summaries = [summary for summary in all_summaries if
                           summary._.name != selected_document._.name]
    remaining_summary_names = [summary._.name for summary in remaining_summaries]

    selected_summary_names = sidebar_placeholder_to.multiselect(
        'Comparison TO:',
        remaining_summary_names,
        remaining_summary_names,
    )
    selected_summaries = []
    for summary_name in selected_summary_names:
        summary_index = remaining_summary_names.index(summary_name)
        selected_summaries.append(remaining_summaries[summary_index])
    return selected_document, selected_summaries


def show_main(example):
    # Get user input

    semantic_sim_type = st.sidebar.radio(
        "Semantic similarity type:",
        ["Contextual embedding", "Static embedding"]
    )
    semantic_sim_threshold = st.sidebar.slider(
        "Semantic similarity threshold:",
        min_value=MIN_SEMANTIC_SIM_THRESHOLD,
        max_value=1.0,
        step=0.1,
        value=0.2,
    )
    semantic_sim_top_k = st.sidebar.slider(
        "Semantic similarity top-k:",
        min_value=1,
        max_value=MAX_SEMANTIC_SIM_TOP_K,
        step=1,
        value=10,
    )

    document, summaries = select_comparison(example)
    layout = st.sidebar.radio("Layout:", ["Vertical", "Horizontal"]).lower()
    scroll = True
    gray_out_stopwords = st.sidebar.checkbox(label="Gray out stopwords", value=True)

    # Gather data
    try:
        lexical_alignments = [
            NGramAlignerCap.decode(
                example.data[
                    Identifier(NGramAlignerCap.__name__)(
                        columns=[
                            f'preprocessed_{document._.column}',
                            f'preprocessed_{summary._.column}',
                        ]
                    )
                ])[0]
            for summary in summaries
        ]
        lexical_alignments = [
            {k: [(pair[0], int(pair[1])) for pair in v]
             for k, v in d.items()}
            for d in lexical_alignments
        ]
    except KeyError:
        lexical_alignments = NGramAligner().align(document, summaries)

    if semantic_sim_type == "Static embedding":
        try:
            semantic_alignments = [
                StaticEmbeddingAlignerCap.decode(
                    example.data[
                        Identifier(StaticEmbeddingAlignerCap.__name__)(
                            threshold=MIN_SEMANTIC_SIM_THRESHOLD,
                            top_k=MAX_SEMANTIC_SIM_TOP_K,
                            columns=[
                                f'preprocessed_{document._.column}',
                                f'preprocessed_{summary._.column}',
                            ]
                        )
                    ])[0]
                for summary in summaries
            ]
        except KeyError:
            semantic_alignments = StaticEmbeddingAligner(
                semantic_sim_threshold,
                semantic_sim_top_k).align(
                document,
                summaries
            )
        else:
            semantic_alignments = [
                filter_alignment(alignment, semantic_sim_threshold, semantic_sim_top_k)
                for alignment in semantic_alignments
            ]
    else:
        try:
            semantic_alignments = [
                BertscoreAlignerCap.decode(
                    example.data[
                        Identifier(BertscoreAlignerCap.__name__)(
                            threshold=MIN_SEMANTIC_SIM_THRESHOLD,
                            top_k=MAX_SEMANTIC_SIM_TOP_K,
                            columns=[
                                f'preprocessed_{document._.column}',
                                f'preprocessed_{summary._.column}',
                            ]
                        )
                    ])[0]
                for summary in summaries
            ]
        except KeyError:
            semantic_alignments = BertscoreAligner(semantic_sim_threshold,
                                                   semantic_sim_top_k).align(document,
                                                                             summaries)
        else:
            semantic_alignments = [
                filter_alignment(alignment, semantic_sim_threshold, semantic_sim_top_k)
                for alignment in semantic_alignments
            ]
    
    MainView(
        document,
        summaries,
        semantic_alignments,
        lexical_alignments,
        layout,
        scroll,
        gray_out_stopwords,
    ).show(height=720)

def _check_model_preds_exist(selected_tenant, selected_batch, batch):
    preds = batch["chats"][0].get("preds") or {}
    models_to_run = [
        model_name
        for model_name in models.keys()
        if model_name not in preds
    ]
    if models_to_run:
        _apply_models(batch, selected_batch, selected_tenant, models_to_run)


def _apply_models(batch, selected_batch, selected_tenant, models_to_run):
    with st.spinner("Running models on batch, this will take some time"):
        databricks.apply_models(selected_tenant, selected_batch, batch, models_to_run)


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    nlp = get_nlp()

    Spacy.encode = _spacy_encode
    Spacy.decode = _spacy_decode
    rg_spacy = Spacy(nlp=nlp)

    selected_batch, selected_tenant = annotation.render_pick_tenant_and_batch()

    if selected_batch:
        batch = annotation.fetch_batch_chats(selected_tenant, selected_batch)
        _check_model_preds_exist(selected_tenant, selected_batch, batch)
        if st.sidebar.button("Rerun all models on batch"):
            _apply_models(selected_tenant, selected_batch, batch, models.keys())

        sidebar_placeholder_from = st.sidebar.empty()
        sidebar_placeholder_to = st.sidebar.empty()

        dataset_size = len(batch['chats'])
        query = st.number_input(f"Index (Size: {dataset_size}):", value=0, min_value=0,
                                max_value=dataset_size - 1)

        if query is not None:
            example = retrieve(batch, query)
            if example:
                st.text(example.id)
                show_main(example)

