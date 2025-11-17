import os
from typing import Any

import autogen
from autogen.cache import Cache

from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant

# ---------- LLM CONFIG (no external OAI_CONFIG_LIST file needed) ----------

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o",  # or another model you have access to
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
        }
    ],
    "timeout": 120,
    "temperature": 0,
}

# ---------- FINNHUB / OTHER API KEYS (from config_api_keys file) ----------

CONFIG_API_KEYS_PATH = os.path.join(os.path.dirname(__file__), "config_api_keys")
if os.path.exists(CONFIG_API_KEYS_PATH):
    register_keys_from_json(CONFIG_API_KEYS_PATH)
else:
    print(f"[test.py] Warning: config_api_keys file not found at {CONFIG_API_KEYS_PATH}")


# ---------- SingleAssistant subclass that RETURNS the final message ----------

class SingleAssistantWithReturn(SingleAssistant):
    """
    Same as SingleAssistant, but chat(...) returns a clean summary string
    instead of the full ChatResult object.
    """

    def chat(self, message: str, use_cache: bool = False, **kwargs) -> str:
        from typing import Any

        from autogen.agentchat.conversable_agent import ChatResult

        with Cache.disk() as cache:
            final = self.user_proxy.initiate_chat(
                self.assistant,
                message=message,
                cache=cache if use_cache else None,
                **kwargs,
            )

        # --- Extract only the useful text ---
        text: Any = None

        # 1) If it's a ChatResult, prefer its summary
        if isinstance(final, ChatResult):
            if getattr(final, "summary", None):
                text = final.summary
            elif getattr(final, "chat_history", None):
                # fall back to last message in the history
                last = final.chat_history[-1]
                if isinstance(last, dict) and "content" in last:
                    text = last["content"]
                else:
                    text = str(last)
            else:
                text = str(final)

        # 2) If it's a plain dict with "content"
        elif isinstance(final, dict) and "content" in final:
            text = final["content"]

        # 3) Fallback
        else:
            text = str(final)

        # Reset agents like the original implementation
        self.reset()
        return str(text)


# ---------- Public API for server.py ----------

DEFAULT_COMPANY = "APPLE"


def get_company_analysis(company: str = DEFAULT_COMPANY) -> str:
    """
    Run the FinRobot SingleAssistant workflow for the given company and
    return the analysis text.
    """
    assistant = SingleAssistantWithReturn(
        "Market_Analyst",
        llm_config,
        human_input_mode="NEVER",
    )

    prompt = (
        f"Use all the tools provided to retrieve information available for {company} "
        f"as of {get_current_date()}. Analyze the positive developments and potential "
        f"concerns of {company} with 2–4 of the most important factors for each, "
        f"kept concise. Most factors should be inferred from company-related news. "
        f"Then make a rough prediction (e.g., up/down by 2–3%) of the {company} stock "
        f"price movement for next week, and provide a summary analysis to support "
        f"your prediction."
    )

    result_text = assistant.chat(prompt)
    return result_text


if __name__ == "__main__":
    print(get_company_analysis())



