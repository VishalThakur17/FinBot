# app.py
import streamlit as st

from tutorial_runners import (
    run_beginner_agent_rag_qa,
    run_beginner_agent_annual_report,
    run_advanced_agent_trade_strategist,
)

st.set_page_config(page_title="FinRobot Tutorials UI", layout="wide")

st.title("FinRobot Tutorials")
st.write("Run tutorial workflows from a simple UI instead of Jupyter Notebooks.")

# --- Tutorial selection ---

TUTORIALS = {
    "Beginner – Agent RAG QA": run_beginner_agent_rag_qa,
    "Beginner – Agent Annual Report": run_beginner_agent_annual_report,
    "Advanced – Agent Trade Strategist": run_advanced_agent_trade_strategist,
    # Add more mappings as you create runner functions
}

tutorial_name = st.selectbox("Choose a tutorial to run:", list(TUTORIALS.keys()))

# We can show some context-specific inputs here if needed
user_question = st.text_input(
    "Question / prompt to send to the agent:",
    value="How is MSFT's 2023 income? Provide some analysis.",
)

run_button = st.button("Run Tutorial")

if run_button:
    runner = TUTORIALS[tutorial_name]

    with st.spinner(f"Running: {tutorial_name} ..."):
        try:
            # Call the selected runner function
            result = runner(user_question)
        except TypeError:
            # Some runners may not take the question param – call with no args
            result = runner()

    st.subheader("Output")
    if isinstance(result, str):
        st.text(result)
    else:
        st.write(result)
