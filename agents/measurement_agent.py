import json
import os
from typing import Optional

from dotenv import load_dotenv
from core.llm_client import AgentsLLM

load_dotenv()


class MeasurementAgent:
    def __init__(self, scenario) -> None:
        self.agent_hist = ""
        self.information = ""
        self.scenario = scenario

        self.llm = AgentsLLM(
            model=os.getenv("MEASUREMENT_MODEL_ID"),
            apiKey=os.getenv("MEASUREMENT_API_KEY"),
            baseUrl=os.getenv("MEASUREMENT_BASE_URL"),
        )

        self.pipe = None
        self.reset()

    def build_dialog_input(self, question: str) -> str:
        return (
            f"Here is the history of the dialogue:\n"
            f"{self.agent_hist if self.agent_hist else '[No previous dialogue]'}\n\n"
            f"Here was the doctor's measurement request:\n{question}\n\n"
            f"Please respond as the measurement reader.\n"
            f'Respond only in the format: "RESULTS: [results here]"'
        )

    def inference_measurement(self, question: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.build_dialog_input(question)},
        ]

        answer = self.llm.think(messages, temperature=0)
        if answer is None:
            return None

        self.agent_hist += f"Doctor: {question}\nMeasurement: {answer}\n\n"
        return answer

    def system_prompt(self) -> str:
        base = (
            "You are a measurement reader who responds with medical test results. "
            'You must respond only in the exact format: "RESULTS: [results here]". '
            "Use only the information provided to you. "
            "If the requested test results are not present in your available data, "
            'respond with: "RESULTS: NORMAL READINGS".'
        )

        presentation = (
            "\n\nBelow is all of the information you have:\n"
            f"{json.dumps(self.information, ensure_ascii=False, indent=2) if isinstance(self.information, dict) else self.information}\n\n"
            "Only return the requested measurement/test result."
        )

        return base + presentation

    def add_hist(self, hist_str: str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_info