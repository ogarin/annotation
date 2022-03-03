import streamlit as st
import pandas as pd
import json
import os
from databricks import load_chats, load_metadata

DATA_DIR = "data"
ANNOTATIONS_PATH = f"{DATA_DIR}/saved_annotations.json"
METADATA_PATH = f"{DATA_DIR}/chat_metadata.csv"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

@st.cache
def get_metadata():
    if os.path.exists(METADATA_PATH):
        return pd.read_csv(METADATA_PATH)

    chat_metadata_df = pd.DataFrame(load_metadata())
    chat_metadata_df.to_csv(METADATA_PATH, index=False)
    return chat_metadata_df

@st.cache
def get_dummy_chats():
    return [
        {"uid": str(cidx),
         "chat_text": "\n".join(f"Hello, this is sent #{idx}" for idx in range(cidx + 5))}
        for cidx in range(5)
    ]

@st.cache
def get_some_real_chats():
    metadata = get_metadata()
    chats_md = metadata[
        metadata.n_turns.between(10, 30)
    ].head(10).to_dict('records')
    return load_chats(chats_md)

# min_turns = st.sidebar.slider(0
#     "Min Turns:",
#     min_value=1,
#     max_value=100,
#     step=1,
#     value=10,
# )
#
# max_turns = st.sidebar.slider(
#     "Max Turns:",
#     min_value=1,
#     max_value=100,
#     step=1,
#     value=50,
# )

def load_annotations():
    if 'annotations' in st.session_state:
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

def show_chat(chat):
    st.button(
        "Clear Annotations",
        key="clear_anns_top",
        on_click=clear_annotations, args=(chat["uid"],)
    )
    if chat["uid"] not in st.session_state.annotations:
        st.session_state.annotations[chat["uid"]] = {}
    for lidx, line in enumerate(chat['chat_text'].splitlines()):
        lidx = str(lidx)
        # col1, col2 = st.columns((1, 20))
        st.checkbox(
            line,
            key=f"radio_{chat['uid']}_{lidx}",
            value=st.session_state.annotations[chat["uid"]].get(lidx),
            on_change=update_annotation,
            args=(
                st.session_state.annotations[chat["uid"]],
                lidx
            )
        )
        # col2.write(line)

    st.button(
        "Clear Annotations",
        key="clear_anns_bottom",
        on_click=clear_annotations, args=(chat["uid"],)
    )

load_annotations()

sample = st.sidebar.radio("Load Chats", ["Dummy chats", "Wiley sample"], index=0)

chats = get_some_real_chats() if "Wiley" in sample else get_dummy_chats()
dataset_size = len(chats)
selected_idx = st.number_input(f"Index:", value=0, min_value=0, max_value=dataset_size - 1)

if selected_idx is not None:
    show_chat(chats[selected_idx])
