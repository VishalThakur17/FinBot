# tutorial_runners.py
import autogen
from finrobot.agents.workflow import SingleAssistantRAG


def run_beginner_agent_rag_qa(
    question: str = "How is MSFT's 2023 income? Provide some analysis.",
):
    """
    Runs the beginner 'agent_rag_qa' tutorial logic and returns the assistant's reply
    as a plain string for the UI.
    """
    # LLM config – same idea as in the notebook, just made into Python code
    llm_config = {
        "config_list": autogen.config_list_from_json(
            "OAI_CONFIG_LIST",  # path relative to where you run the app
            filter_dict={"model": ["gpt-4-0125-preview"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }

    assistant = SingleAssistantRAG(
        "Data_Analyst",
        llm_config,
        human_input_mode="NEVER",
        retrieve_config={
            "task": "qa",
            "vector_db": None,  # as in the notebook comment
            "docs_path": [
                "report/Microsoft_Annual_Report_2023.pdf",
            ],
            "chunk_token_size": 1000,
            "get_or_create": True,
            "collection_name": "msft_analysis",
            "must_break_at_empty_line": False,
        },
        rag_description="Retrieve content from MSFT's 2023 annual report for detailed question answering.",
    )

    # SingleAssistantRAG.chat(...) runs the conversation and returns the final reply
    reply = assistant.chat(question)
    return str(reply)


def run_beginner_agent_annual_report(
    question: str = "Summarize the key points from MSFT's 2023 annual report.",
):
    """
    Example runner for tutorials_beginner/agent_annual_report.ipynb.
    Adapt the body to match the notebook's code.
    """
    # TODO: copy the relevant notebook code here:
    # - llm_config
    # - whatever assistant / workflow class it's using
    # - call .chat(...) or equivalent
    return "Not implemented yet. Replace this with the notebook logic."


def run_advanced_agent_trade_strategist(
    symbol: str = "AAPL",
):
    """
    Example runner for tutorials_advanced/agent_trade_strategist.ipynb.
    Adapt this using the notebook code.
    """
    # TODO: replicate the notebook steps using the finrobot APIs
    return f"Trade strategist analysis for {symbol} (not implemented yet)."
