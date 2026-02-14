from __future__ import annotations
import streamlit as st


def info_box(title: str, items: list[str]):
    with st.container(border=True):
        st.subheader(title)
        for it in items:
            st.write(it)


def section(title: str, desc: str | None = None):
    st.markdown(f"## {title}")
    if desc:
        st.caption(desc)