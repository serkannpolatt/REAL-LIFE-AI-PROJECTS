import streamlit as st
import pandas as pd
from streamlit_chat import message
from data_processing import DataProcessor
from chatbot import FinancialChatbot
import os
from warnings import filterwarnings

filterwarnings("ignore")

st.title("Financial Advisor AI Chatbot")


@st.cache_resource
def get_data(alloc_file, financ_file):
    df_alloc = pd.read_csv(alloc_file)
    df_financial = pd.read_csv(financ_file)
    processor = DataProcessor(df_alloc, df_financial)
    if not os.path.exists(processor.save_path):
        os.makedirs(processor.save_path)
    processor.save_processed_data()
    return df_alloc, df_financial


with st.sidebar:
    data_ready = False
    api_key = st.text_input("OpenAI API key", type="password")
    alloc_file = st.file_uploader("Upload target allocations file")
    financ_file = st.file_uploader("Upload financial data file")
    if alloc_file and financ_file:
        try:
            df_alloc, df_financial = get_data(alloc_file, financ_file)
            data_ready = True
            st.success("Data is ready to use!")
        except Exception as e:
            st.error(f"Data processing error: {e}")

if data_ready:
    bot = FinancialChatbot(df_alloc, df_financial, api_key=api_key)
    eval_metrics = []
    if "history" not in st.session_state:
        st.session_state.history = []
    user_input = st.chat_input("You: ")
    if user_input:
        st.session_state.history.append({"message": user_input, "is_user": True})
        with st.spinner("Generating response..."):
            response_text, tokens_used, cost = bot.generate_response(user_input)
            eval_score, _ = bot.evaluate_response_by_relevancy(
                user_input, response_text
            )
            eval_metrics.append(eval_score)
            with st.sidebar:
                eval_metrics_mean = (
                    round(sum(eval_metrics) / len(eval_metrics), 2)
                    if eval_metrics
                    else 0
                )
                st.info(
                    f"Last message relevancy score: {eval_score}\nAverage relevancy score: {eval_metrics_mean}"
                )
        st.session_state.history.append({"message": response_text, "is_user": False})
        st.session_state.user_input = ""
        for idx, chat in enumerate(st.session_state.history, 1):
            message(chat["message"], is_user=chat["is_user"], key=str(idx))
