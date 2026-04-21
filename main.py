

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

    # 先统一复用同一个 LLM 池
    agent_prefixes = ["MASTER", "PATIENT", "INTAKE", "SPECIALIST", "MEASUREMENT"]

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
        "specialist_llm": llms["specialist"],
        "measurement_llm": llms["measurement"],
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

    # 这里复用 MasterAgent 类，分别挂不同 profile
    router_agent = agents.MasterAgent(
        sender="MASTER_ROUTER",
        profile=profiles.MASTER_ROUTER_PROFILE,
        scenario=scenario,
        case_store=case_store,
        llm_client=master_llm,
        message_store=message_store,
        max_discuss_turns=1,
        img_processing=img_processing,
    )

    single_doctor_agent = agents.MasterAgent(
        sender="MASTER_DOCTOR",
        profile=profiles.MASTER_SINGLE_DOCTOR_PROFILE,
        scenario=scenario,
        case_store=case_store,
        llm_client=master_llm,
        message_store=message_store,
        max_discuss_turns=max_discuss_turns,
        img_processing=img_processing,
    )

    coordinator_agent = agents.MasterAgent(
        sender="MASTER_COORDINATOR",
        profile=profiles.MASTER_COORDINATOR_PROFILE,
        scenario=scenario,
        case_store=case_store,
        llm_client=master_llm,
        message_store=message_store,
        max_discuss_turns=1,
        img_processing=img_processing,
    )

    return patient_agent, intake_agent, router_agent, single_doctor_agent, coordinator_agent


def build_specialist_agents(
    scenario,
    case_store: CaseStore,
    message_store: MessageStore,
    specialist_llm,
    specialties: list[str],
    img_processing: bool,
):
    specialist_agents = {}

    for specialty in specialties:
        specialist_agents[specialty] = agents.MasterAgent(
            sender=f"{specialty.upper()}_SPECIALIST",
            profile=profiles.build_specialist_profile(specialty),
            scenario=scenario,
            case_store=case_store,
            llm_client=specialist_llm,
            message_store=message_store,
            max_discuss_turns=1,
            img_processing=img_processing,
        )

    return specialist_agents


def create_case_windows(message_store: MessageStore, specialties: list[str] | None = None) -> dict[str, str]:
    intake_window_id = "intake_patient_window"
    router_window_id = "router_window"
    single_doctor_window_id = "single_doctor_window"
    coordinator_window_id = "coordinator_window"

    message_store.create_window(
        window_id=intake_window_id,
        title="Intake-Patient Dialogue",
        participants=["INTAKE", "PATIENT"],
    )

    message_store.create_window(
        window_id=router_window_id,
        title="Router Decision",
        participants=["MASTER_ROUTER"],
    )

    message_store.create_window(
        window_id=single_doctor_window_id,
        title="Single Doctor Reasoning",
        participants=["MASTER_DOCTOR", "MEASUREMENT"],
    )

    message_store.create_window(
        window_id=coordinator_window_id,
        title="Coordinator Summary",
        participants=["MASTER_COORDINATOR"],
    )

    specialist_window_ids = {}
    specialties = specialties or []
    for specialty in specialties:
        wid = f"{specialty}_specialist_window"
        specialist_window_ids[specialty] = wid
        message_store.create_window(
            window_id=wid,
            title=f"{specialty.capitalize()} Specialist Discussion",
            participants=[f"{specialty.upper()}_SPECIALIST", "MEASUREMENT"],
        )

    return {
        "intake_window_id": intake_window_id,
        "router_window_id": router_window_id,
        "single_doctor_window_id": single_doctor_window_id,
        "coordinator_window_id": coordinator_window_id,
        "specialist_window_ids": specialist_window_ids,
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


def parse_json_from_message(content: str) -> dict:
    content = content.strip()
    return json.loads(content)


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

        if intake_msg.metadata.get("intake_done"):
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


def run_router_phase(router_agent, window_id: str, sleep_time: float) -> dict:
    router_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content="Review the structured case and decide whether to continue on the single-doctor path or initiate specialist discussion.",
    )

    router_msg = router_agent.inference(window_id)
    print(f"Router: {router_msg.content}")
    time.sleep(sleep_time)

    return parse_json_from_message(router_msg.content)


def handle_measurement_request(request_text: str) -> str:
    # 这里先给你一个测试版占位实现
    # 后面你接 MeasurementAgent / scenario data 时再替换
    if "REQUEST TEST:" in request_text:
        test_name = request_text.replace("REQUEST TEST:", "").strip()
        return f"RESULTS: {test_name} = NORMAL READINGS"
    return "RESULTS: NORMAL READINGS"


def run_single_doctor_phase(
    single_doctor_agent,
    window_id: str,
    max_discuss_turns: int,
    sleep_time: float,
) -> str:
    single_doctor_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content="Review the structured case and continue diagnosis. If needed, request tests. If ready, output DIAGNOSIS READY.",
    )

    final_diagnosis = ""

    for inf_id in range(max_discuss_turns):
        msg = single_doctor_agent.inference(window_id)
        print(f"SingleDoctor [{inf_id + 1}/{max_discuss_turns}]: {msg.content}")
        time.sleep(sleep_time)

        if msg.content.startswith("REQUEST TEST:"):
            result_text = handle_measurement_request(msg.content)
            single_doctor_agent.message_store.append_message(
                window_id=window_id,
                sender="MEASUREMENT",
                content=result_text,
            )
            print(f"Measurement: {result_text}")
            time.sleep(sleep_time)
            continue

        if msg.content.startswith("DIAGNOSIS READY:"):
            final_diagnosis = msg.content.replace("DIAGNOSIS READY:", "").strip()
            break

    return final_diagnosis


def run_specialist_phase(
    specialist_agents: dict,
    specialist_window_ids: dict[str, str],
    max_discuss_turns: int,
    sleep_time: float,
) -> list[dict]:
    specialist_outputs = []

    for specialty, specialist_agent in specialist_agents.items():
        window_id = specialist_window_ids[specialty]

        specialist_agent.message_store.append_message(
            window_id=window_id,
            sender="SYSTEM",
            content="Review the structured case, provide specialist evidence, and recommend additional tests if needed.",
        )

        last_content = ""
        for inf_id in range(max_discuss_turns):
            msg = specialist_agent.inference(window_id)
            last_content = msg.content
            print(f"{specialty.capitalize()} Specialist [{inf_id + 1}/{max_discuss_turns}]: {msg.content}")
            time.sleep(sleep_time)

            if "THAT IS ALL" in msg.content:
                break

            # 简单处理专家建议检查：主程序代为执行 measurement
            json_part = msg.content.replace("THAT IS ALL", "").strip()
            try:
                data = json.loads(json_part)
            except Exception:
                data = {}

            for test_name in data.get("recommended_measurements", []):
                result_text = f"RESULTS: {test_name} = NORMAL READINGS"
                specialist_agent.message_store.append_message(
                    window_id=window_id,
                    sender="MEASUREMENT",
                    content=result_text,
                )
                print(f"Measurement for {specialty}: {result_text}")
                time.sleep(sleep_time)

        json_part = last_content.replace("THAT IS ALL", "").strip()
        try:
            specialist_outputs.append(json.loads(json_part))
        except Exception:
            specialist_outputs.append({
                "specialist": specialty,
                "opinion": "",
                "supporting_evidence": [],
                "recommended_measurements": [],
                "confidence": "low",
            })

    return specialist_outputs


def run_coordinator_phase(coordinator_agent, window_id: str, sleep_time: float) -> dict:
    coordinator_agent.message_store.append_message(
        window_id=window_id,
        sender="SYSTEM",
        content="Summarize all available specialist evidence and output the final diagnosis JSON.",
    )

    msg = coordinator_agent.inference(window_id)
    print(f"Coordinator: {msg.content}")
    time.sleep(sleep_time)

    return parse_json_from_message(msg.content)


def run_single_case(
    scenario_id: int,
    scenario,
    patient_llm,
    intake_llm,
    master_llm,
    specialist_llm,
    runtime_config: dict,
):
    sleep_time = runtime_config["sleep_time"]
    max_intake_turns = runtime_config["max_intake_turns"]
    max_discuss_turns = runtime_config["max_discuss_turns"]
    img_processing = runtime_config["img_processing"]
    inf_type = runtime_config["inf_type"]

    case_store = CaseStore(str(scenario_id))
    message_store = MessageStore()

    # 先建基础窗口
    window_ids = create_case_windows(message_store)

    patient_agent, intake_agent, router_agent, single_doctor_agent, coordinator_agent = build_case_agents(
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

    router_result = run_router_phase(
        router_agent=router_agent,
        window_id=window_ids["router_window_id"],
        sleep_time=sleep_time,
    )

    final_result = None

    if router_result["next_action"] == "single_doctor":
        final_diagnosis = run_single_doctor_phase(
            single_doctor_agent=single_doctor_agent,
            window_id=window_ids["single_doctor_window_id"],
            max_discuss_turns=max_discuss_turns,
            sleep_time=sleep_time,
        )
        final_result = {
            "final_diagnosis": final_diagnosis,
            "path": "single_doctor",
        }

    elif router_result["next_action"] == "initiate_specialist_discussion":
        specialties = router_result.get("required_specialists", [])
        # 重新创建专家窗口
        specialist_window_ids = {}
        for specialty in specialties:
            wid = f"{specialty}_specialist_window"
            specialist_window_ids[specialty] = wid
            message_store.create_window(
                window_id=wid,
                title=f"{specialty.capitalize()} Specialist Discussion",
                participants=[f"{specialty.upper()}_SPECIALIST", "MEASUREMENT"],
            )

        specialist_agents = build_specialist_agents(
            scenario=scenario,
            case_store=case_store,
            message_store=message_store,
            specialist_llm=specialist_llm,
            specialties=specialties,
            img_processing=img_processing,
        )

        specialist_outputs = run_specialist_phase(
            specialist_agents=specialist_agents,
            specialist_window_ids=specialist_window_ids,
            max_discuss_turns=max_discuss_turns,
            sleep_time=sleep_time,
        )

        # 暂时把专家输出写到 final_report 前的中间结果里，后面你可以单独扩 CaseStore
        case_store.data["specialist_outputs"] = specialist_outputs

        final_result = run_coordinator_phase(
            coordinator_agent=coordinator_agent,
            window_id=window_ids["coordinator_window_id"],
            sleep_time=sleep_time,
        )

    else:
        raise ValueError(f"Unknown next_action: {router_result['next_action']}")

    print("\n[Final Result]")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))

    return {
        "case_store": case_store,
        "message_store": message_store,
        "router_result": router_result,
        "final_result": final_result,
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
    specialist_llm = system_state["specialist_llm"]

    runtime_config = system_state["runtime_config"]

    for scenario_id in range(max_cases):
        scenario = scenario_loader.get_scenario(id=scenario_id)

        run_single_case(
            scenario_id=scenario_id,
            scenario=scenario,
            patient_llm=patient_llm,
            intake_llm=intake_llm,
            master_llm=master_llm,
            specialist_llm=specialist_llm,
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
        help="Maximum number of doctor/specialist reasoning turns.",
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