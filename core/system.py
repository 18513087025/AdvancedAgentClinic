from dataclasses import dataclass
from typing import Any
import json
from core.scenario_loader import ScenarioLoader


@dataclass
class SystemContext:
    scenario_loader: Any
    llm_pool: dict[str, Any]
    runtime_config: dict[str, Any]
    max_cases: int

    def get_llm(self, role: str):
        return self.llm_pool[role]


def initialize_system(
    dataset: str,
    llms: dict[str, Any],
    inf_type: str,
    max_intake_turns: int,
    max_discuss_turns: int,
    img_processing: bool,
    sleep_time: float,
    num_scenarios: int | None = None,
) -> SystemContext:
    scenario_loader = ScenarioLoader(dataset)

    if num_scenarios is None:
        max_cases = scenario_loader.num_scenarios
    else:
        max_cases = min(num_scenarios, scenario_loader.num_scenarios)

    llm_pool = {
        "master": llms["master"],
        "patient": llms["patient"],
        "intake": llms["intake"],
        "specialist": llms["specialist"],
        "measurement": llms["measurement"],
        "evaluator": llms["evaluator"],
    }

    runtime_config = {
        "inference": {
            "type": inf_type,
            "max_intake_turns": max_intake_turns,
            "max_discuss_turns": max_discuss_turns,
        },
        "system": {
            "sleep_time": sleep_time,
            "img_processing": img_processing,
        },
        "data": {
            "dataset": dataset,
            "num_scenarios": num_scenarios,
        },
    }

    return SystemContext(
        scenario_loader=scenario_loader,
        llm_pool=llm_pool,
        runtime_config=runtime_config,
        max_cases=max_cases,
    )





def extract_structured_intake(intake_agent, window_id: str) -> dict:
    dialogue_history = intake_agent.message_store.get_window_text(window_id)

    base_prompt = f"""
    Now, based on the full dialogue history, organize the patient information.
    Return a valid JSON object with the following schema:

    {{
        "patient_profile": {{}},
        "physical_exam": {{}},
        "labs": {{}},
        "imaging": {{}},
        "pathology": {{}}
    }}

    Strict requirements:
    - Output ONLY JSON
    - No explanation
    - No markdown
    - No extra text

    Dialogue history:
    {dialogue_history}
    """

    messages = [
        {"role": "system", "content": intake_agent.system_prompt()},
        {"role": "user", "content": base_prompt},
    ]

    max_retries = 2
    answer = None

    for retry_id in range(max_retries + 1):
        try:
            answer = intake_agent.llm_client.think(messages, temperature=0)
        except Exception as e:
            if retry_id == max_retries:
                raise RuntimeError(
                    f"[{intake_agent.sender}] Intake structured extraction failed: {e}"
                )
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response failed. "
                    "Please strictly follow the output format and return VALID JSON ONLY."
                ),
            })
            continue

        if answer is None:
            if retry_id == max_retries:
                raise ValueError(
                    f"[{intake_agent.sender}] Intake structured extraction returned None"
                )
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was empty. "
                    "Please strictly follow the output format and return VALID JSON ONLY."
                ),
            })
            continue

        if not isinstance(answer, str):
            answer = str(answer)

        try:
            structured = json.loads(answer)
            return structured
        except json.JSONDecodeError:
            if retry_id == max_retries:
                raise ValueError(
                    f"[{intake_agent.sender}] Intake structured extraction failed: invalid JSON\n{answer}"
                )
            messages.append({
                "role": "assistant",
                "content": answer,
            })
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Please strictly follow the output format and return ONLY a valid JSON object."
                ),
            })

    raise ValueError("Intake structured extraction failed after retries")



def write_structured_intake_to_case_store(case_store, structured: dict) -> None:
    mapping = {
        "patient_profile": case_store.set_patient_profile,
        "physical_exam": case_store.update_physical_exam,
        "labs": case_store.update_labs,
        "imaging": case_store.update_imaging,
        "pathology": case_store.update_pathology,
    }

    for key, func in mapping.items():
        value = structured.get(key)
        if not isinstance(value, dict):
            value = {}
        func(value)


def finalize_intake_phase(intake_agent, case_store, window_id: str, save_path: str | None = None) -> dict:
    structured = extract_structured_intake(intake_agent, window_id)
    write_structured_intake_to_case_store(case_store, structured)

    if save_path is not None:
        case_store.save_json(save_path)

    return structured