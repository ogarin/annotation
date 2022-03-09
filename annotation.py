import streamlit as st
import pandas as pd
import json
import os
from databricks import (
    load_chats,
    load_metadata,
    load_batch,
    load_tenants,
    create_batch,
    get_tenant_temp_dir
)

DATA_DIR = "data"
ANNOTATIONS_PATH = f"{DATA_DIR}/saved_annotations.json"
METADATA_PATH = f"{DATA_DIR}/chat_metadata.csv"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


@st.cache
def get_metadata(tenant_name):
    if os.path.exists(METADATA_PATH):
        chat_metadata_df = pd.read_csv(METADATA_PATH)
        if chat_metadata_df.loc[0, "chat_file"].startswith(get_tenant_temp_dir(tenant_name)):
            return chat_metadata_df

    chat_metadata_df = pd.DataFrame(load_metadata(tenant_name))
    chat_metadata_df.to_csv(METADATA_PATH, index=False)
    return chat_metadata_df


@st.cache
def get_dummy_chats():
    return [
        {
            "uid": str(cidx),
            "chat_text": "\n".join(
                f"Hello, this is turn #{idx}" for idx in range(cidx + 5)
            ),
        }
        for cidx in range(5)
    ]


@st.cache
def get_some_real_chats(tenant_name, batch_name):
    metadata = get_metadata(tenant_name)
    chats_md = metadata[metadata.n_turns.between(10, 30)].head(10).to_dict("records")
    return load_chats(chats_md)


def load_annotations():
    if "annotations" in st.session_state:
        return

    st.session_state.annotations = {}
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH) as fp:
            st.session_state.annotations = json.load(fp)


def save_annotations():
    with open(ANNOTATIONS_PATH, "w") as fp:
        json.dump(st.session_state.annotations, fp)


def clear_annotations(chat_idx):
    del st.session_state.annotations[chat_idx]
    save_annotations()


def update_annotation(annotations, idx):
    annotations[idx] = not annotations.get(idx)
    save_annotations()


def render_chat(chat):
    st.button(
        "Clear Annotations",
        key="clear_anns_top",
        on_click=clear_annotations,
        args=(chat["uid"],),
    )
    if chat["uid"] not in st.session_state.annotations:
        st.session_state.annotations[chat["uid"]] = {}
    for lidx, line in enumerate(chat["chat_text"].splitlines()):
        lidx = str(lidx)
        # col1, col2 = st.columns((1, 20))
        st.checkbox(
            line,
            key=f"radio_{chat['uid']}_{lidx}",
            value=st.session_state.annotations[chat["uid"]].get(lidx),
            on_change=update_annotation,
            args=(st.session_state.annotations[chat["uid"]], lidx),
        )
        # col2.write(line)

    st.button(
        "Clear Annotations",
        key="clear_anns_bottom",
        on_click=clear_annotations,
        args=(chat["uid"],),
    )


def render_create_new_batch(
    selected_tenant,
):
    new_batch_name = st.text_input("Batch Name")
    batch_size = st.number_input("Batch Size", value=50)
    turn_range = st.slider("Turn Range:", value=(5, 10))
    if st.button("Create"):
        create_batch(selected_tenant, new_batch_name, batch_size, turn_range)


def render_sidebar():
    tenants = load_tenants()
    selected_tenant = st.sidebar.selectbox("Pick Tenant", tenants)

    with st.sidebar.expander("Create New Batch"):
        render_create_new_batch(selected_tenant)

    batchs = load_batch(selected_tenant)
    st.sidebar.header("Batches")
    selected_batch = st.sidebar.radio("Pick Batch", batchs)
    return selected_tenant, selected_batch


def render_annotation_window(selected_tenant, selected_batch):
    chats = get_some_real_chats(selected_tenant, selected_batch)
    dataset_size = len(chats)
    selected_idx = st.number_input(
        f"Index:", value=0, min_value=0, max_value=dataset_size - 1
    )
    if selected_idx is not None:
        with st.container():
            render_chat(chats[selected_idx])


load_annotations()
selected_tenant, selected_batch = render_sidebar()
render_annotation_window(selected_tenant, selected_batch)
