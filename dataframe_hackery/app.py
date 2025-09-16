import os

import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

st.set_page_config(page_title="Chat with Your CSV", layout="wide")
st.title("📊 Chat with Your CSV (PandasAI)")

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.dataframe(df.head())

    # Replace with your API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY is not set. Please add it to your .env file.")

    llm = OpenAI(api_token=api_key)
    sdf = SmartDataframe(df, config={"llm": llm, "verbose": True})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = sdf.chat(prompt)
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    elif hasattr(result, "savefig"):
                        st.pyplot(result)
                    else:
                        st.markdown(str(result))
                except Exception as e:
                    st.error(f"Error: {e}")
