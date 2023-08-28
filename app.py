import streamlit as st
import pandas as pd
import plotly.express as px
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

st.title("pandas-ai streamlit interface")

st.write("A demo interface for [PandasAI](https://github.com/gventuri/pandas-ai)")
st.write(
    "Looking for an example *.csv-file?, check [here](https://gist.github.com/netj/8836201)."
)

if "openai_key" not in st.session_state:
    with st.form("API key"):
        key = st.text_input("OpenAI Key", value="", type="password")
        if st.form_submit_button("Submit"):
            st.session_state.openai_key = key
            st.session_state.prompt_history = []
            st.session_state.df = None

if "openai_key" in st.session_state:
    # Select dataset option
    st.sidebar.subheader("Select Dataset")
    dataset_option = st.sidebar.radio("Choose a dataset:", ["Upload CSV", "Use Example"])

    # Load selected dataset
    if dataset_option == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file. This should be in long format (one datapoint per row).",
            type="csv",
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
    elif dataset_option == "Use Example":
        # Load example dataset
        df = pd.read_csv("link_to_example.csv")  # Replace with the actual link
        st.session_state.df = df

    with st.form("Question"):
        question = st.text_input("Question", value="", type="default")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with st.spinner():
                llm = OpenAI(api_token=st.session_state.openai_key)
                pandas_ai = PandasAI(llm)
                x = pandas_ai.run(st.session_state.df, prompt=question)

                fig = px.scatter(x=x.index, y=x.values)
                st.plotly_chart(fig)

                st.write(x)
                st.session_state.prompt_history.append(question)

    if st.session_state.df is not None:
        st.subheader("Current dataframe:")
        st.write(st.session_state.df)

    st.subheader("Prompt history:")
    st.write(st.session_state.prompt_history)

if st.button("Clear"):
    st.session_state.prompt_history = []
    st.session_state.df = None
