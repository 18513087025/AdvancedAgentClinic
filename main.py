import argparse
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from configs import profiles

from core.system import initialize_system
from core.case_store import CaseStore
from core.message_store import MessageStore
from core.llm_client import AgentsLLM

from agents.patient import PatientAgent
from agents.intake import IntakeAgent
from agents.router import RouterAgent
from agents.measurement import MeasurementAgent
from agents.specialists import SpecialistAgent
from agents.coordinator import CoordinatorAgent


def append_case_store_jsonl(
    case_store: CaseStore,
    path: str = "./output/case_store.jsonl",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(case_store.to_dict(), ensure_ascii=False) + "\n")


def build_case_agents(
    scenario,
    case_store: CaseStore,
    message_store: MessageStore,
    llms: dict,
    max_intake_turns: int,
    max_discuss_turns: int,
    img_processing: bool,
):
    patient_agent = PatientAgent(
        sender="PATIENT",
        profile=profiles.PATIENT_PROFILE,
        scenario=scenario,
        llm_client=llms["patient"],
        message_store=message_store,
        img_processing=img_processing,
    )

    intake_agent = IntakeAgent(
        sender="INTAKE",
        profile=profiles.INTAKE_PROFILE,
        scenario=scenario,
        llm_client=llms["intake"],
        message_store=message_store,
        max_intake_turns=max_intake_turns,
        img_processing=img_processing,
    )

    router_agent = RouterAgent(
        sender="ROUTER",
        profile=profiles.ROUTER_PROFILE,
        scenario=scenario,
        case_store=case_store,
        llm_client=llms["master"],
        message_store=message_store,
        img_processing=img_processing,
    )

    measurement_agent = MeasurementAgent(
        sender="MEASUREMENT",
        profile=profiles.MEASUREMENT_PROFILE,
        scenario=scenario,
        llm_client=llms["measurement"],
        message_store=message_store,
        img_processing=img_processing,
    )

    coordinator_agent = CoordinatorAgent(
        sender="COORDINATOR",
        profile=profiles.COORDINATOR_PROFILE,
        scenario=scenario,
        case_store=case_store,
        llm_client=llms["master"],
        message_store=message_store,
        max_discussion_turns=max_discuss_turns,
        img_processing=img_processing,
    )

    return (
        patient_agent,
        intake_agent,
        router_agent,
        measurement_agent,
        coordinator_agent,
    )


def build_specialist_agents(
    scenario,
    case_store: CaseStore,
    message_store: MessageStore,
    specialist_llm,
    specialties: list[str],
    max_discuss_turns: int,
    img_processing: bool,
):
    specialist_agents = {}

    for specialty in specialties:
        specialist_agents[specialty] = SpecialistAgent(
            sender=f"{specialty.upper()}_SPECIALIST",
            profile=profiles.build_specialist_profile(specialty),
            scenario=scenario,
            case_store=case_store,
            llm_client=specialist_llm,
            message_store=message_store,
            specialty=specialty,
            max_discussion_turns=max_discuss_turns,
            img_processing=img_processing,
        )

    return specialist_agents


def create_case_windows(message_store: MessageStore) -> dict[str, str]:
    intake_window_id = "intake_window"
    discussion_window_id = "discussion_window"

    message_store.create_window(
        window_id=intake_window_id,
        title="Intake-Patient Dialogue",
        participants=["SYSTEM", "INTAKE", "PATIENT"],
    )

    message_store.create_window(
        window_id=discussion_window_id,
        title="Specialist Discussion",
        participants=["SYSTEM", "ROUTER", "MEASUREMENT"],
    )

    return {
        "intake_window_id": intake_window_id,
        "discussion_window_id": discussion_window_id,
    }


def add_specialists_to_discussion_window(
    message_store: MessageStore,
    window_id: str,
    specialties: list[str],
) -> None:
    window = message_store.windows[window_id]
    for specialty in specialties:
        participant = f"{specialty.upper()}_SPECIALIST"
        if participant not in window.participants:
            window.participants.append(participant)


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

        if intake_msg.metadata.get("done"):
            print(f"\n{'=' * 30}")
            print("[Intake Complete]")
            print(f"{'=' * 30}")
            case_store.update_patient_info(intake_msg.metadata["structured_data"])
            return

        patient_msg = patient_say(
            inf_type=inf_type,
            patient_agent=patient_agent,
            window_id=window_id,
        )
        print(f"Patient [{progress}%]: {patient_msg.content}")
        time.sleep(sleep_time)


def run_router_phase(router_agent, window_id: str, sleep_time: float) -> dict:
    router_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content=(
            "Review the structured case, assess complexity, and determine which "
            "specialists should join the discussion."
        ),
    )

    router_msg = router_agent.inference(window_id)
    print(f"Router: {router_msg.content}")
    print(f"\n{'=' * 30}")
    print("[Routing Complete]")
    print(f"{'=' * 30}")
    time.sleep(sleep_time)

    return router_msg.metadata["structured_data"]


def run_measurement_step(
    measurement_agent,
    case_store: CaseStore,
    window_id: str,
    test_name: str,
    sleep_time: float,
    requester: str = "UNKNOWN",
) -> dict:
    measurement_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content=(
            f"A test has been requested by {requester}. "
            f"Please process this request and return the measurement result for: {test_name}"
        ),
    )

    measurement_msg = measurement_agent.inference(window_id)
    if measurement_msg is None:
        return {}

    result = measurement_msg.metadata.get("structured_data", {})
    case_store.update_measurement(test_name, result)

    if isinstance(result, dict):
        category = result.get("category")
        patch = result.get("patient_info_update")

        if isinstance(category, str) and isinstance(patch, dict):
            try:
                case_store.update_patient_info_from_measurement(category, patch)
            except ValueError:
                pass

    print(f"Measurement ({requester} requested): {measurement_msg.content}")
    print("[Measurement Complete]")
    time.sleep(sleep_time)

    return result


def build_discussion_order(
    specialist_agents: dict,
) -> list[tuple[str, object]]:
    speakers = []

    for specialty, agent in specialist_agents.items():
        speakers.append((specialty, agent))

    return speakers


def prime_discussion_context(
    specialist_agents: dict,
    window_id: str,
) -> None:
    shared_prompt = (
        "This is a multi-agent specialist discussion. "
        "You must read and respond to previous messages from other specialists. "
        "Do not ignore prior specialist comments. "
        "You may do one of the following in each turn:\n"
        "1. Provide discussion / critique / agreement / disagreement.\n"
        "2. Request a test using the exact format: REQUEST TEST: [test_name]\n"
        "3. Output final JSON only when your view is stable and sufficiently supported.\n"
        "Discussion quality matters: explicitly reference or react to other specialists when relevant."
    )

    if specialist_agents:
        any_agent = next(iter(specialist_agents.values()))
        any_agent.message_store.append_message(
            window_id=window_id,
            sender="SYSTEM",
            content=shared_prompt,
        )


def run_discussion_phase(
    specialist_agents: dict,
    measurement_agent,
    case_store: CaseStore,
    window_id: str,
    max_discuss_turns: int,
    sleep_time: float,
) -> dict:
    specialist_outputs = {}
    finished_specialists: set[str] = set()

    prime_discussion_context(
        specialist_agents=specialist_agents,
        window_id=window_id,
    )

    speakers = build_discussion_order(
        specialist_agents=specialist_agents,
    )

    for round_id in range(max_discuss_turns):
        print(f"\n{'=' * 30}")
        print(f"[Discussion Round {round_id + 1}]")
        print(f"{'=' * 30}")

        anyone_spoke = False

        for speaker_name, speaker_agent in speakers:
            if speaker_name in finished_specialists:
                continue

            msg = speaker_agent.inference(window_id)
            if msg is None:
                continue

            anyone_spoke = True

            print(f"{speaker_name.capitalize()} Specialist [Round {round_id + 1}]: {msg.content}")
            time.sleep(sleep_time)

            output_type = msg.metadata.get("output_type")

            if output_type == "request_test":
                test_name = msg.metadata["structured_data"]["test_request"]
                run_measurement_step(
                    measurement_agent=measurement_agent,
                    case_store=case_store,
                    window_id=window_id,
                    test_name=test_name,
                    sleep_time=sleep_time,
                    requester=speaker_name,
                )
                continue

            if output_type == "discussion":
                continue

            if output_type == "final_opinion":
                final_opinion = msg.metadata["structured_data"]
                specialist_outputs[speaker_name] = final_opinion
                case_store.update_specialist_opinion(speaker_name, final_opinion)
                finished_specialists.add(speaker_name)
                print(f"\n[{speaker_name.capitalize()} Specialist Complete]")
                continue

        if not anyone_spoke:
            break

        if len(finished_specialists) == len(specialist_agents):
            break

    print(f"\n{'=' * 30}")
    print("[Specialist Discussion Complete]")
    print(f"{'=' * 30}")

    return specialist_outputs


def run_coordinator_phase(
    coordinator_agent,
    case_store: CaseStore,
    window_id: str,
    max_discuss_turns: int,
    sleep_time: float,
) -> dict:
    if case_store.to_dict().get("final_report"):
        return case_store.to_dict()["final_report"]

    coordinator_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content=(
            "All specialist discussions have ended. "
            "Now summarize the structured case and finalized specialist opinions, "
            "then output the final diagnosis JSON."
        ),
    )

    for inf_id in range(max_discuss_turns):
        msg = coordinator_agent.inference(window_id)
        if msg is None:
            continue

        print(f"Coordinator [{inf_id + 1}/{max_discuss_turns}]: {msg.content}")
        time.sleep(sleep_time)

        output_type = msg.metadata.get("output_type")

        if output_type == "discussion":
            continue

        if output_type == "final_report":
            final_report = msg.metadata["structured_data"]
            case_store.update_final_report(final_report)

            print(f"\n{'=' * 30}")
            print("[Coordinator Complete]")
            print(f"{'=' * 30}")

            return final_report

    return {}


def run_single_case(
    scenario_id: int,
    scenario,
    llms: dict,
    runtime_config: dict,
):
    inference_config = runtime_config["inference"]
    system_config = runtime_config["system"]

    sleep_time = system_config["sleep_time"]
    img_processing = system_config["img_processing"]

    inf_type = inference_config["type"]
    max_intake_turns = inference_config["max_intake_turns"]
    max_discuss_turns = inference_config["max_discuss_turns"]

    case_store = CaseStore(str(scenario_id))
    message_store = MessageStore()

    window_ids = create_case_windows(message_store)

    (
        patient_agent,
        intake_agent,
        router_agent,
        measurement_agent,
        coordinator_agent,
    ) = build_case_agents(
        scenario=scenario,
        case_store=case_store,
        message_store=message_store,
        llms=llms,
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

    router_result = run_router_phase(
        router_agent=router_agent,
        window_id=window_ids["discussion_window_id"],
        sleep_time=sleep_time,
    )
    case_store.update_triage(router_result)

    specialties = router_result.get("required_specialists", [])

    add_specialists_to_discussion_window(
        message_store=message_store,
        window_id=window_ids["discussion_window_id"],
        specialties=specialties,
    )

    specialist_agents = build_specialist_agents(
        scenario=scenario,
        case_store=case_store,
        message_store=message_store,
        specialist_llm=llms["specialist"],
        specialties=specialties,
        max_discuss_turns=max_discuss_turns,
        img_processing=img_processing,
    )

    specialist_outputs = run_discussion_phase(
        specialist_agents=specialist_agents,
        measurement_agent=measurement_agent,
        case_store=case_store,
        window_id=window_ids["discussion_window_id"],
        max_discuss_turns=max_discuss_turns,
        sleep_time=sleep_time,
    )

    print("\n[Specialist Opinions]")
    print(json.dumps(specialist_outputs, ensure_ascii=False, indent=2))

    final_result = run_coordinator_phase(
        coordinator_agent=coordinator_agent,
        case_store=case_store,
        window_id=window_ids["discussion_window_id"],
        max_discuss_turns=max_discuss_turns,
        sleep_time=sleep_time,
    )

    print(f"\n{'=' * 30}")
    print("[Final Result]")
    print(f"{'=' * 30}")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))

    append_case_store_jsonl(case_store, "./output/case_store.jsonl")

    return {
        "case_store": case_store,
        "message_store": message_store,
        "router_result": router_result,
        "specialist_outputs": specialist_outputs,
        "final_result": final_result,
    }


def main(
    inf_type: str = "llm",
    dataset: str = "MedQA",
    num_scenarios: Optional[int] = 1,
    max_intake_turns: int = 5,
    max_discuss_turns: int = 5,
    img_processing: bool = False,
):
    llms = {
        name: AgentsLLM(
            os.getenv("MODEL_ID"),
            os.getenv("API_KEY"),
            os.getenv("BASE_URL"),
        )
        for name in [
            "master",
            "patient",
            "intake",
            "specialist",
            "measurement",
            "evaluator",
        ]
    }

    system = initialize_system(
        dataset=dataset,
        llms=llms,
        inf_type=inf_type,
        max_intake_turns=max_intake_turns,
        max_discuss_turns=max_discuss_turns,
        img_processing=img_processing,
        sleep_time=1.0,
        num_scenarios=num_scenarios,
    )

    os.makedirs("./output", exist_ok=True)
    with open("./output/case_store.jsonl", "w", encoding="utf-8") as f:
        pass

    scenario_loader = system.scenario_loader
    max_cases = system.max_cases
    runtime_config = system.runtime_config

    for scenario_id in range(max_cases):
        scenario = scenario_loader.get_scenario(id=scenario_id)

        run_single_case(
            scenario_id=scenario_id,
            scenario=scenario,
            llms=llms,
            runtime_config=runtime_config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent Medical Diagnosis Simulation System"
    )

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
        help="Enable asking about medical images when supported.",
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
        help="Maximum number of specialist/coordinator reasoning turns.",
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