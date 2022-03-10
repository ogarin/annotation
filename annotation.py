import streamlit as st
import pandas as pd
import json
import os
from functools import partial
from databricks import (
    load_chats,
    load_metadata,
    load_batch,
    load_tenants,
    create_batch,
    get_tenant_temp_dir,
    fetch_batch_meta,
    upload_annotation,
    fetch_annotation,
)

DATA_DIR = "data"


if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


def _get_annotation_path(tenant_name, batch_name):
    return f"{DATA_DIR}/{tenant_name}_{batch_name}.anno"


def _meta_data_path(tenant_name):
    return f"{DATA_DIR}/{tenant_name}_metadata.csv"


@st.cache
def get_metadata(tenant_name):
    metadata_path = _meta_data_path(tenant_name)
    if os.path.exists(metadata_path):
        chat_metadata_df = pd.read_csv(metadata_path)
        if chat_metadata_df.loc[0, "chat_file"].startswith(
            get_tenant_temp_dir(tenant_name)
        ):
            return chat_metadata_df

    chat_metadata_df = pd.DataFrame(load_metadata(tenant_name))
    chat_metadata_df.to_csv(metadata_path, index=False)
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
def fetch_batch_chats(tenant_name, batch_name):
    metadata = get_metadata(tenant_name)
    chats_md = metadata.to_dict("records")
    batch_meta = fetch_batch_meta(tenant_name, batch_name)
    chats_md_filterd = [
        chat for chat in chats_md if chat["uid"] in batch_meta["chat_uids"]
    ]
    return {"chats": load_chats(chats_md_filterd), "batch_meta": batch_meta}


def load_annotations(selected_tenant, selected_batch):
    annotation = {}
    try:
        with open(
            _get_annotation_path(selected_tenant, selected_batch),
            "r",
        ) as tfile:
            annotation = json.load(tfile)
    except:
        pass
    st.session_state["annotations"] = annotation


def load_annotation_from_remote(selected_tenant, selected_batch):
    st.session_state.annotations = fetch_annotation(selected_tenant, selected_batch)
    save_annotations(selected_tenant, selected_batch)


def upload_annotation_to_remote(selected_tenant, selected_batch):
    upload_annotation(selected_tenant, selected_batch, st.session_state.annotations)


def save_annotations(selected_tenant, selected_batch):
    with open(
        _get_annotation_path(selected_tenant, selected_batch),
        "w",
    ) as tfile:
        return json.dump(st.session_state.annotations, tfile)


def clear_sample_annotations(selected_tenant, selected_batch, chat_idx):
    del st.session_state.annotations[chat_idx]
    save_annotations(selected_tenant, selected_batch)


def update_annotation(selected_tenant, selected_batch, annotations, idx, current_value):
    annotations[idx] = not current_value
    save_annotations(selected_tenant, selected_batch)


def render_chat(chat):
    st.button(
        "Clear Annotations",
        key="clear_anns_top",
        on_click=clear_sample_annotations,
        args=(selected_tenant, selected_batch, chat["uid"]),
    )
    if chat["uid"] not in st.session_state.annotations:
        st.session_state.annotations[chat["uid"]] = {
            idx: False for idx, _ in enumerate(chat["chat_text"].splitlines())
        }
    for lidx, line in enumerate(chat["chat_text"].splitlines()):
        lidx = str(lidx)
        # col1, col2 = st.columns((1, 20))
        st.checkbox(
            line,
            key=f"radio_{chat['uid']}_{lidx}",
            value=st.session_state.annotations[chat["uid"]].get(lidx),
            on_change=update_annotation,
            args=(
                selected_tenant,
                selected_batch,
                st.session_state.annotations[chat["uid"]],
                lidx,
                st.session_state.annotations[chat["uid"]].get(lidx)
            ),
        )
        # col2.write(line)

    st.button(
        "Clear Annotations",
        key="clear_anns_bottom",
        on_click=clear_sample_annotations,
        args=(chat["uid"],),
    )


def render_create_new_batch(
    selected_tenant,
):
    new_batch_name = st.text_input("Batch Name")
    batch_size = st.number_input("Batch Size", value=50)
    turn_range = st.slider("Turn Range:", value=(5, 10))
    if st.button("Create"):
        with st.spinner(f"Creating Batch {new_batch_name}, Please wait..."):
            with st.spinner("Creating new Batch..."):
                create_batch(selected_tenant, new_batch_name, batch_size, turn_range)


def render_sidebar():
    tenants = load_tenants()
    selected_tenant = st.sidebar.selectbox("Pick Tenant", tenants)
    batchs = load_batch(selected_tenant)
    selected_batch = st.sidebar.selectbox("Pick Batch", [""] + batchs)
    with st.sidebar.expander("Create New Batch"):
        render_create_new_batch(selected_tenant)

    if selected_batch:
        if st.sidebar.button("Fetch Annotation"):
            with st.spinner("Fetch Annotation..."):
                load_annotation_from_remote(selected_tenant, selected_batch)

        if st.sidebar.button("Upload Annotation"):
            with st.spinner("Uploading Annotation..."):
                upload_annotation_to_remote(selected_tenant, selected_batch)

    return selected_tenant, selected_batch


def render_annotation_window(selected_tenant, selected_batch):
    with st.container():
        with st.spinner(f"Loading Samples from batch: {selected_batch}"):
            batch = fetch_batch_chats(selected_tenant, selected_batch)
        dataset_size = len(batch["chats"])
        st.header(
            f'Batch: {batch["batch_meta"]["name"]} Created: {batch["batch_meta"]["create_date"]}'
        )
        selected_idx = st.number_input(
            f"Index:", value=0, min_value=0, max_value=dataset_size - 1
        )
        if selected_idx is not None:
            with st.container():
                render_chat(batch["chats"][selected_idx])


if __name__ == "__main__":
    selected_tenant, selected_batch = render_sidebar()
    if selected_batch:
        load_annotations(selected_tenant, selected_batch)
        render_annotation_window(selected_tenant, selected_batch)
