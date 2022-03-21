import streamlit as st
import pandas as pd
import json
import os
from common import DATA_DIR, get_annotation_local_path
from encryption_utils import decrypt_by_sample, encrypt_by_sample
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


def load_annotations(selected_tenant, selected_batch, annotation_type):
    annotations = {}
    try:
        with open(
            get_annotation_local_path(selected_tenant, selected_batch, annotation_type),
            "r",
        ) as tfile:
            annotations = json.load(tfile)
    except:
        pass

    st.session_state["annotations"] = annotations


def load_annotation_from_remote(selected_tenant, selected_batch, annotation_type):
    st.session_state.annotations = fetch_annotation(
        selected_tenant, selected_batch, annotation_type
    )
    save_annotations(selected_tenant, selected_batch, annotation_type)


def upload_annotation_to_remote(selected_tenant, selected_batch, annotation_type):
    upload_annotation(
        selected_tenant, selected_batch, annotation_type, st.session_state.annotations
    )


def save_annotations(selected_tenant, selected_batch, annotation_type):
    with open(
        get_annotation_local_path(selected_tenant, selected_batch, annotation_type), "w"
    ) as tfile:
        return json.dump(st.session_state.annotations, tfile)


def clear_sample_annotations(
    selected_tenant, selected_batch, annotation_type, chat_idx
):
    del st.session_state.annotations[chat_idx]
    save_annotations(selected_tenant, selected_batch, annotation_type)


def update_annotation(selected_tenant, selected_batch, annotation_type, chat, line_idx):
    chat_uid = chat["uid"]
    curr_annotation = {}
    if 'enc_content' in st.session_state.annotations[chat_uid]:
        curr_annotation = decrypt_by_sample(chat, st.session_state.annotations[chat_uid]['enc_content'])

    if st.session_state[f"radio_{line_idx}"]:
        curr_annotation[line_idx] = st.session_state.get(f"summary_{line_idx}", "")
    else:
        curr_annotation.pop(line_idx, None)
    st.session_state.annotations[chat_uid] = {
        "uid": chat_uid,
        "enc_content": encrypt_by_sample(chat, curr_annotation),
    }
    save_annotations(selected_tenant, selected_batch, annotation_type)


def update_free_text_annotation(chat):
    summary_text = st.session_state["summary_text_area"]
    print(f'saving annotation: {chat["uid"]}')
    enc_summary = encrypt_by_sample(chat, {"summary": summary_text})
    st.session_state.annotations[chat["uid"]] = enc_summary
    save_annotations(selected_tenant, selected_batch, annotation_type)


def render_chat_v2(chat):
    st.text("Chat: " + chat["uid"])
    chat_col, summary_col = st.columns(2)

    if chat["uid"] not in st.session_state.annotations:
        st.session_state.annotations[chat["uid"]] = {}

    with chat_col:
        for lidx, line in enumerate(chat["chat_text"].splitlines()):
            st.text(line)

    with summary_col:
        summary_doc = None
        enc_summary = st.session_state.annotations[chat["uid"]]
        if enc_summary:
            summary_doc = decrypt_by_sample(chat, enc_summary)

        current_summary = summary_doc["summary"] if summary_doc else ""
        st.text_area(
            "Summary",
            value=current_summary,
            key="summary_text_area",
            on_change=update_free_text_annotation,
            args=(chat,),
        )


def render_chat(chat):
    st.text("Chat: " + chat["uid"])
    st.button(
        "Clear Annotations",
        key="clear_anns_top",
        on_click=clear_sample_annotations,
        args=(selected_tenant, selected_batch, annotation_type, chat["uid"]),
    )

    if chat["uid"] not in st.session_state.annotations:
        st.session_state.annotations[chat["uid"]] = {}

    if 'enc_content' in st.session_state.annotations[chat["uid"]]:
        curr_annotations = decrypt_by_sample(
            chat, st.session_state.annotations[chat["uid"]]["enc_content"]
        )
    else:
        curr_annotations = {}

    for lidx, line in enumerate(chat["chat_text"].splitlines()):
        lidx = str(lidx)

        if st.checkbox(
            line,
            key=f"radio_{lidx}",
            value=lidx in curr_annotations,
            on_change=update_annotation,
            args=(selected_tenant, selected_batch, annotation_type, chat, lidx),
        ):
            st.text_input(
                "Next Summary Sentence:",
                key=f"summary_{lidx}",
                value=curr_annotations.get(lidx, ""),
                on_change=update_annotation,
                args=(selected_tenant, selected_batch, annotation_type, chat, lidx),
            )

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


def render_annotation_type():
    return st.sidebar.selectbox("Annotation Types", ["Segments", "Free Text"])


def render_pick_tenant_and_batch():
    annotation_type = render_annotation_type()
    tenants = load_tenants()
    selected_tenant = st.sidebar.selectbox("Pick Tenant", tenants)
    batchs = load_batch_names(selected_tenant)
    selected_batch = st.sidebar.selectbox("Pick Batch", [""] + batchs)
    return selected_batch, selected_tenant, annotation_type


def render_sidebar():
    selected_batch, selected_tenant, annotation_type = render_pick_tenant_and_batch()
    with st.sidebar.expander("Create New Batch"):
        render_create_new_batch(selected_tenant)

    if selected_batch:
        if st.sidebar.button("Fetch Annotation"):
            with st.spinner("Fetch Annotation..."):
                load_annotation_from_remote(
                    selected_tenant, selected_batch, annotation_type
                )

        if st.sidebar.button("Upload Annotation"):
            with st.spinner("Uploading Annotation..."):
                upload_annotation_to_remote(
                    selected_tenant, selected_batch, annotation_type
                )

    return selected_tenant, selected_batch, annotation_type


def render_annotation_window(selected_tenant, selected_batch, annotation_type):
    with st.container():
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
                if annotation_type == "Segments":
                    render_chat(batch["chats"][selected_idx])
                elif annotation_type == "Free Text":
                    render_chat_v2(batch["chats"][selected_idx])


if __name__ == "__main__":
    selected_tenant, selected_batch, annotation_type = render_sidebar()
    if selected_batch:
        load_annotations(selected_tenant, selected_batch, annotation_type)
        render_annotation_window(selected_tenant, selected_batch, annotation_type)
