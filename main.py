import argparse
import json
import time
from typing import Optional
import os

from dotenv import load_dotenv
load_dotenv()

from core import agents
from core.case_store import CaseStore
from core.message_store import MessageStore
from core.llm_client import AgentsLLM
from configs import profiles

from utils.scenario_loader import ScenarioLoader
from utils.parsing import is_intake_complete


def initialize_system(
    dataset: str,
    num_scenarios: Optional[int] = None,
    inf_type: str = "llm",
    max_intake_turns: int = 5,
    max_discuss_turns: int = 5,
    img_processing: bool = False,
    sleep_time: float = 1.0,
) -> dict:
    scenario_loader = ScenarioLoader(dataset)

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    max_cases = min(num_scenarios, scenario_loader.num_scenarios)

    agent_prefixes = ["MASTER", "PATIENT", "INTAKE"]

    llms = {
        name.lower(): AgentsLLM(
            os.getenv("MODEL_ID"),
            os.getenv("API_KEY"),
            os.getenv("BASE_URL"),
        )
        for name in agent_prefixes
    }

    runtime_config = {
        "inf_type": inf_type,
        "max_intake_turns": max_intake_turns,
        "max_discuss_turns": max_discuss_turns,
        "img_processing": img_processing,
        "sleep_time": sleep_time,
        "dataset": dataset,
        "num_scenarios": num_scenarios,
    }

    return {
        "scenario_loader": scenario_loader,
        "max_cases": max_cases,
        "llms": llms,
        "master_llm": llms["master"],
        "patient_llm": llms["patient"],
        "intake_llm": llms["intake"],
        "runtime_config": runtime_config,
    }


def build_case_agents(
    scenario,
    case_store: CaseStore,
    message_store: MessageStore,
    patient_llm,
    intake_llm,
    master_llm,
    max_intake_turns: int,
    max_discuss_turns: int,
    img_processing: bool,
):
    patient_agent = agents.PatientAgent(
        sender="PATIENT",
        profile=profiles.PATIENT_PROFILE,
        scenario=scenario,
        llm_client=patient_llm,
        message_store=message_store,
        img_processing=img_processing,
    )

    intake_agent = agents.IntakeAgent(
        sender="INTAKE",
        profile=profiles.INTAKE_PROFILE,
        scenario=scenario,
        llm_client=intake_llm,
        message_store=message_store,
        max_intake_turns=max_intake_turns,
        img_processing=img_processing,
    )

    master_agent = agents.MasterAgent(
        sender="MASTER",
        profile=profiles.MASTER_PROFILE,
        scenario=scenario,
        case_store=case_store,
        llm_client=master_llm,
        message_store=message_store,
        max_discuss_turns=max_discuss_turns,
        img_processing=img_processing,
    )

    return patient_agent, intake_agent, master_agent


def create_case_windows(message_store: MessageStore) -> dict[str, str]:
    intake_window_id = "intake_patient_window"
    master_window_id = "master_window"

    message_store.create_window(
        window_id=intake_window_id,
        title="Intake-Patient Dialogue",
        participants=["INTAKE", "PATIENT"],
    )

    message_store.create_window(
        window_id=master_window_id,
        title="Master Reasoning",
        participants=["MASTER"],
    )

    return {
        "intake_window_id": intake_window_id,
        "master_window_id": master_window_id,
    }


def patient_say(
    inf_type: str,
    patient_agent,
    window_id: str,
):
    if inf_type == "human_patient":
        patient_text = input("\nPatient: ")
        return patient_agent.message_store.append_message(
            window_id=window_id,
            sender="PATIENT",
            content=patient_text,
        )
    return patient_agent.inference(window_id)


def run_intake_phase(
    intake_agent,
    patient_agent,
    case_store: CaseStore,
    window_id: str,
    inf_type: str,
    max_intake_turns: int,
    sleep_time: float,
) -> None:
    intake_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content="Please begin the consultation. Ask the patient your first question.",
    )

    for inf_id in range(max_intake_turns):
        progress = int(((inf_id + 1) / max_intake_turns) * 100)

        intake_msg = intake_agent.inference(window_id)
        print(f"Intake [{progress}%]: {intake_msg.content}")
        time.sleep(sleep_time)

        if is_intake_complete(intake_msg.content):
            intake_agent.finalize_intake(case_store, window_id)
            return

        patient_msg = patient_say(
            inf_type=inf_type,
            patient_agent=patient_agent,
            window_id=window_id,
        )
        print(f"Patient [{progress}%]: {patient_msg.content}")
        time.sleep(sleep_time)

    intake_agent.finalize_intake(case_store, window_id)


def run_master_phase(
    master_agent,
    window_id: str,
    max_discuss_turns: int,
    sleep_time: float,
):
    master_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content="Review the structured case and decide the next step.",
    )

    last_msg = None

    for inf_id in range(max_discuss_turns):
        progress = int(((inf_id + 1) / max_discuss_turns) * 100)

        if inf_id == max_discuss_turns - 1:
            master_agent.message_store.append_message(
                window_id=window_id,
                sender="SYSTEM",
                content="This is the final discussion turn. If you have enough evidence, provide your final diagnosis.",
            )

        master_msg = master_agent.inference(window_id)
        last_msg = master_msg
        print(f"Master [{progress}%]: {master_msg.content}")
        time.sleep(sleep_time)

        if "[MASTER DONE]" in master_msg.content:
            break

    return last_msg


def run_single_case(
    scenario_id: int,
    scenario,
    patient_llm,
    intake_llm,
    master_llm,
    runtime_config: dict,
):
    sleep_time = runtime_config["sleep_time"]
    max_intake_turns = runtime_config["max_intake_turns"]
    max_discuss_turns = runtime_config["max_discuss_turns"]
    img_processing = runtime_config["img_processing"]
    inf_type = runtime_config["inf_type"]

    case_store = CaseStore(str(scenario_id))
    message_store = MessageStore()

    window_ids = create_case_windows(message_store)

    patient_agent, intake_agent, master_agent = build_case_agents(
        scenario=scenario,
        case_store=case_store,
        message_store=message_store,
        patient_llm=patient_llm,
        intake_llm=intake_llm,
        master_llm=master_llm,
        max_intake_turns=max_intake_turns,
        max_discuss_turns=max_discuss_turns,
        img_processing=img_processing,
    )

    print(f"\n{'=' * 80}")
    print(f"Scenario {scenario_id}")
    print(f"Examiner info: {scenario.examiner_info}")
    print(f"{'=' * 80}\n")

    run_intake_phase(
        intake_agent=intake_agent,
        patient_agent=patient_agent,
        case_store=case_store,
        window_id=window_ids["intake_window_id"],
        inf_type=inf_type,
        max_intake_turns=max_intake_turns,
        sleep_time=sleep_time,
    )

    print("\n[Structured Case]")
    print(json.dumps(case_store.to_dict(), ensure_ascii=False, indent=2))

    master_msg = run_master_phase(
        master_agent=master_agent,
        window_id=window_ids["master_window_id"],
        max_discuss_turns=max_discuss_turns,
        sleep_time=sleep_time,
    )

    return {
        "case_store": case_store,
        "message_store": message_store,
        "master_message": master_msg,
    }


def main(
    inf_type: str = "llm",
    dataset: str = "MedQA",
    num_scenarios: Optional[int] = None,
    max_intake_turns: int = 5,
    max_discuss_turns: int = 5,
    img_processing: bool = False,
):
    system_state = initialize_system(
        dataset=dataset,
        num_scenarios=num_scenarios,
        inf_type=inf_type,
        max_intake_turns=max_intake_turns,
        max_discuss_turns=max_discuss_turns,
        img_processing=img_processing,
        sleep_time=1.0,
    )

    scenario_loader = system_state["scenario_loader"]
    max_cases = system_state["max_cases"]

    patient_llm = system_state["patient_llm"]
    intake_llm = system_state["intake_llm"]
    master_llm = system_state["master_llm"]

    runtime_config = system_state["runtime_config"]

    for scenario_id in range(max_cases):
        scenario = scenario_loader.get_scenario(id=scenario_id)

        run_single_case(
            scenario_id=scenario_id,
            scenario=scenario,
            patient_llm=patient_llm,
            intake_llm=intake_llm,
            master_llm=master_llm,
            runtime_config=runtime_config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Medical Diagnosis Simulation System")

    parser.add_argument(
        "--inf_type",
        type=str,
        choices=["llm", "human_patient"],
        default="llm",
        help="Type of inference: 'llm' for fully automated, 'human_patient' to manually control the patient.",
    )
    parser.add_argument(
        "--img_processing",
        action="store_true",
        help="Enable the doctor to request medical images (if available in the dataset).",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=1,
        help="Number of clinical cases to run.",
    )
    parser.add_argument(
        "--max_intake_turns",
        type=int,
        default=5,
        help="Maximum number of intake-patient turns.",
    )
    parser.add_argument(
        "--max_discuss_turns",
        type=int,
        default=5,
        help="Maximum number of master discussion turns.",
    )
    parser.add_argument(
        "--agent_dataset",
        type=str,
        default="MedQA",
        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext", "MIMICIV"],
        help="Dataset to use for simulation.",
    )
    args = parser.parse_args()

    main(
        inf_type=args.inf_type,
        dataset=args.agent_dataset,
        num_scenarios=args.num_scenarios,
        max_intake_turns=args.max_intake_turns,
        max_discuss_turns=args.max_discuss_turns,
        img_processing=args.img_processing,
    )