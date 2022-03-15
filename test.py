import pandas as pd
import streamlit as st

df = pd.DataFrame([
    {
        "text": f"This is sent {idx}",
        "idx": 10000 + idx,
        "label": "",
    }
    for idx in range(10)
])

st.markdown(""" <style>
[data-testid="stHorizontalBlock"] label {display:none;}
[data-testid="stHorizontalBlock"] [data-baseweb="select"] > div > div {
    padding-bottom:0;
    padding-top:0;
}
</style> """, unsafe_allow_html=True)

labels = st.text_input("Labels (comma separated)")

for rowidx, row in df.iterrows():
    # st.text_input(row.text, key=f"label_{rowidx}")
    col1, col2 = st.columns((1,4))
    with col1:
        st.selectbox("", key=f"label_{rowidx}", options=[""] + [
            l.strip() for l in labels.split(",")
        ])
    # col1.selectbox("", key=f"label_{rowidx}", options=[
    #     '', 'Issue', 'Step', 'Solution'
    # ])
    col2.markdown(row.text)