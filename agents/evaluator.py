import os
from typing import Optional

from dotenv import load_dotenv
from core.llm_client import AgentsLLM

load_dotenv()


class EvaluatorAgent:
    def __init__(self) -> None:
        self.llm = AgentsLLM(
            model=os.getenv("MODERATOR_MODEL_ID"),
            apiKey=os.getenv("MODERATOR_API_KEY"),
            baseUrl=os.getenv("MODERATOR_BASE_URL"),
        )

    def system_prompt(self) -> str:
        return (
            "You are a strict medical evaluator.\n"
            "Determine whether the two diagnoses refer to the same disease.\n"
            "Consider synonyms and variations.\n"
            'Respond with ONLY one word: "Yes" or "No".\n'
            "No explanation."
        )

    def evaluate(self, diagnosis: str, correct_diagnosis: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Correct diagnosis:\n{correct_diagnosis}\n\n"
                    f"Doctor diagnosis:\n{diagnosis}\n\n"
                    "Are they the same disease?"
                ),
            },
        ]

        answer = self.llm.think(messages, temperature=0)
        if answer is None:
            return None

        answer = answer.strip().lower()
        if "yes" in answer:
            return "yes"
        if "no" in answer:
            return "no"
        return answer