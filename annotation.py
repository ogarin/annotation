import streamlit as st
import pandas as pd
import json
import os
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


def _meta_data_path(tenant_name):
    return f"{DATA_DIR}/{tenant_name}_metadata.csv"

def get_annotation_config_path():
    return f"{DATA_DIR}/annotation_config.json"


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
    annotation_config = {"turn_categories": []}
    try:
        with open(get_annotation_config_path(), "r") as tfile:
            annotation_config = json.load(tfile)
    except:
        pass
    st.session_state["annotation_config"] = annotation_config
    st.session_state["turn_categories_str"] = ",".join(annotation_config["turn_categories"])

    annotations = {}
    try:
        with open(get_annotation_local_path(selected_tenant, selected_batch), "r") as tfile:
            annotations = json.load(tfile)
    except:
        pass

    st.session_state["annotations"] = annotations


def load_annotation_from_remote(selected_tenant, selected_batch):
    st.session_state.annotations = fetch_annotation(selected_tenant, selected_batch)
    save_annotations(selected_tenant, selected_batch)


def upload_annotation_to_remote(selected_tenant, selected_batch):
    upload_annotation(selected_tenant, selected_batch, st.session_state.annotations)


def save_annotations(selected_tenant, selected_batch):
    with open(get_annotation_local_path(selected_tenant, selected_batch), "w") as tfile:
        return json.dump(st.session_state.annotations, tfile)


def clear_sample_annotations(selected_tenant, selected_batch, chat_idx):
    del st.session_state.annotations[chat_idx]
    save_annotations(selected_tenant, selected_batch)


def update_annotation_config():
    st.session_state.annotation_config["turn_categories"] = [
        cat.strip()
        for cat in st.session_state.turn_categories_str.split(",")
    ]
    with open(get_annotation_config_path(), "w") as tfile:
        return json.dump(st.session_state.annotation_config, tfile)

def update_annotation(selected_tenant, selected_batch, chat_uid, line_idx):
    st.session_state.annotations[chat_uid][line_idx] = st.session_state[f"turn_label_{line_idx}"]
    print(st.session_state.annotations[chat_uid])
    save_annotations(selected_tenant, selected_batch)


def render_chat(chat):
    st.markdown(""" <style>
    [data-testid="stHorizontalBlock"] label {display:none;}
    [data-testid="stHorizontalBlock"] [data-baseweb="select"] > div > div {
        padding-bottom:0;
        padding-top:0;
    }
    </style> """, unsafe_allow_html=True)

    st.text("Chat: " + chat["uid"])
    st.button(
        "Clear Annotations",
        key="clear_anns_top",
        on_click=clear_sample_annotations,
        args=(selected_tenant, selected_batch, chat["uid"]),
    )

    if chat["uid"] not in st.session_state.annotations:
        st.session_state.annotations[chat["uid"]] = {}

    turn_cats = [""] + st.session_state.annotation_config["turn_categories"]
    for lidx, line in enumerate(chat["chat_text"].splitlines()):
        lidx = str(lidx)

        col1, col2 = st.columns((1, 4))
        with col1:
            st.selectbox(
                "<LABEL HIDDEN>",
                key=f"turn_label_{lidx}",
                index=turn_cats.index(st.session_state.annotations[chat["uid"]].get(lidx, "")),
                options=turn_cats,
                on_change=update_annotation,
                args=(
                    selected_tenant,
                    selected_batch,
                    chat["uid"],
                    lidx
                ),
            )

        with col2:
            st.markdown(line)

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


def render_pick_tenant_and_batch():
    tenants = load_tenants()
    selected_tenant = st.sidebar.selectbox("Pick Tenant", tenants)
    batchs = load_batch_names(selected_tenant)
    selected_batch = st.sidebar.selectbox("Pick Batch", [""] + batchs)
    return selected_batch, selected_tenant


def render_sidebar():
    selected_batch, selected_tenant = render_pick_tenant_and_batch()
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
        st.text_input(
            "Labels (comma separated)",
            key="turn_categories_str",
            on_change=update_annotation_config
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
