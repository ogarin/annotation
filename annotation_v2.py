import json

import streamlit as st
import hydralit as hy
import pandas as pd
from datetime import datetime
from annotated_text import annotated_text
import re

import encryption_utils
from common import DATA_DIR, get_annotation_local_path
from databricks import (
    load_chats,
    load_metadata,
    load_batch_names,
    load_tenants,
    create_batch,
    get_tenant_temp_dir,
    fetch_batch_meta,
    upload_annotation,
    fetch_annotation,
)
from annotation import get_metadata


def load_annotations(selected_tenant, selected_batch):
    annotations = {}
    try:
        with open(
            get_annotation_local_path(selected_tenant, selected_batch), "r"
        ) as tfile:
            annotations = json.load(tfile)
    except:
        pass

    st.session_state["annotations"] = annotations


def save_annotations(selected_tenant, selected_batch):
    with open(get_annotation_local_path(selected_tenant, selected_batch), "w") as tfile:
        return json.dump(st.session_state.annotations, tfile)


def _meta_data_path(tenant_name):
    return f"{DATA_DIR}/{tenant_name}_metadata.csv"


def get_annotation_config_path():
    return f"{DATA_DIR}/annotation_config.json"


app = hy.HydraApp(title="Summarization Annotation")


def parse_summary(text):
    segments = []
    current_segment = []
    current_region = None
    tokens = re.split(r"[ \n]", text)
    for token in tokens:
        match = re.match(r"^<(\/?)([A-Za-z]+)>$", token)
        if match:
            is_close = match.group(1)
            region = match.group(2)

            if current_segment:
                segments.append((current_segment, current_region))
                current_segment = []

            if is_close:
                if current_region != region:
                    raise Exception()
                current_region = None
            else:
                if current_region is not None:
                    raise Exception()
                current_region = region
        else:
            current_segment.append(token)
    return segments


def render_chat(chat):
    st.markdown(
        """ <style>
    [data-testid="stHorizontalBlock"] label {display:none;}
    [data-testid="stHorizontalBlock"] [data-baseweb="select"] > div > div {
        padding-bottom:0;
        padding-top:0;
    }
    </style> """,
        unsafe_allow_html=True,
    )

    st.text("Chat: " + chat["uid"])

    if chat["uid"] not in st.session_state.annotations:
        st.session_state.annotations[chat["uid"]] = {}

    # turn_cats = [""] + st.session_state.annotation_config["turn_categories"]
    for line in chat["chat_text"].splitlines():
        st.markdown(line)


@app.addapp(title="Manage Batch")
def create_batch():
    with st.container():
        st.subheader("Batchs")
        st.table(
            pd.DataFrame(
                [
                    {"Name": "Batch1", "Size": 50, "created": datetime.now()},
                    {"Name": "Batch2", "Size": 50, "created": datetime.now()},
                ]
            )
        )


@st.cache
def fetch_batch_chats(tenant_name, batch_name):
    metadata = get_metadata(tenant_name)
    chats_md = metadata.to_dict("records")
    batch_meta = fetch_batch_meta(tenant_name, batch_name)
    chats_md_filterd = [
        chat for chat in chats_md if chat["uid"] in batch_meta["chat_uids"]
    ]
    return {"chats": load_chats(chats_md_filterd), "batch_meta": batch_meta}


def update_annotation(selected_tenant, selected_batch, chat_uid, enc_doc):
    st.session_state.annotations[chat_uid] = enc_doc
    save_annotations(selected_tenant, selected_batch)


@app.addapp(title="Free Text")
def free_text_annotation():
    selected_tenant, selected_batch = None, None
    with st.container():
        possible_tenants = load_tenants()
        selected_tenant = st.selectbox("Select Tenants", options=possible_tenants)
        possible_batches = load_batch_names(selected_tenant)
        selected_batch = st.selectbox("Select Batch", options=possible_batches)

    load_annotations(selected_tenant, selected_batch)
    with st.container():
        chat_col, sum_col = st.columns(2)
        with chat_col:
            batch = fetch_batch_chats(selected_tenant, selected_batch)
            dataset_size = len(batch["chats"])
            st.subheader(
                f'Batch: {batch["batch_meta"]["name"]} Created: {batch["batch_meta"]["create_date"]}'
            )
            selected_idx = st.number_input(
                f"Index:", value=0, min_value=0, max_value=dataset_size - 1
            )
            current_sample = batch["chats"][selected_idx]
            render_chat(current_sample)

        with sum_col:
            st.subheader("Summarization")
            st.text("Please Use <> to mark part of summary, e.g: <issue>....</issue>")

            current_summary = ""
            anno_doc = st.session_state.annotations.get(current_sample["uid"])
            if anno_doc:
                key = encryption_utils.dervice_encrpytion_from_sample(current_sample)
                doc = json.loads(
                    encryption_utils.decrypt_using_sample(key, anno_doc["doc"]).decode(
                        "utf-8"
                    )
                )
                current_summary = ' '.join(f' <{tag}> {text} </{tag}> ' for tag, text in doc["summary"])
            summary = st.text_area(
                "Please Write Here Summary",
                value=current_summary,
            )
            if summary:
                parsed_summary = parse_summary(summary)
                for region, anno in parsed_summary:
                    st.write(f'{anno}: {" ".join(region)}')
                args = [
                    (" ".join(region), anno, "#fea") if anno else " ".join(region)
                    for region, anno in parsed_summary
                ]
                annotated_text(*args)
                doc = {
                    "summary": [
                        {anno: " ".join(text)} for text, anno in parsed_summary
                    ],
                }
                key = encryption_utils.dervice_encrpytion_from_sample(current_sample)
                encrypted_doc = {
                    "uid": current_sample["uid"],
                    "doc": encryption_utils.encrypt_using_sample(
                        key, json.dumps(doc).encode("utf-8")
                    ),
                }
                update_annotation(
                    selected_tenant,
                    selected_batch,
                    current_sample["uid"],
                    encrypted_doc,
                )
    # with st.container():
    #     if st.button('Upload Annotations'):

app.run()
