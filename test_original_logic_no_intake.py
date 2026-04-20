import argparse
import time
from typing import Optional
import os


from core import agents
from core.case_store import CaseStore

from agents.evaluator_agent import DiagnosisEvaluator
from agents.measurement_agent import MeasurementAgent

from core.llm_client import AgentsLLM
from utils.scenario_loader import ScenarioLoader
from utils.parsing import *


from dotenv import load_dotenv
load_dotenv()

def main(
    inf_type: str = "llm",            # 推理模式: "llm" | "human_doctor" | "human_patient"
    dataset: str = "MedQA",          # 数据集: "MedQA" | "MedQA_Ext" | "NEJM" | "NEJM_Ext" | "MIMICIV"
    num_scenarios: Optional[int] = None,  # 运行case数量: None=全部 | int>0=指定数量
    max_infs: int = 20,              # 最大交互轮数: 正整数 (如 10, 20, 30)
    img_processing: bool = False,    # 是否启用影像分析能力: True=允许请求影像 | False=禁止
):
    """
    Main entry for multi-agent medical diagnosis simulation.

    Args:
        inf_type (str):
            推理模式，可选值：
            - "llm": 医生和病人均由 LLM 自动完成
            - "human_doctor": 人类扮演医生，LLM 扮演病人
            - "human_patient": LLM 扮演医生，人类扮演病人

        dataset (str):
            使用的数据集，可选值：
            - "MedQA"
            - "MedQA_Ext"
            - "NEJM"
            - "NEJM_Ext"
            - "MIMICIV"

        num_scenarios (Optional[int]):
            运行的病例数量：
            - None: 使用全部数据
            - 正整数: 仅运行前 N 个 case

        max_infs (int):
            最大对话轮数（doctor-patient interaction turns）：
            - 正整数（推荐 10~30）

        img_processing (bool):
            是否启用影像相关能力：
            - True: 允许询问影像并触发 INTAKE_IMAGE / REQUEST IMAGES
            - False: 禁止涉及影像信息

    """
    
    scenario_loader = ScenarioLoader(dataset)

    total_correct = 0 # 诊断正确的case
    total_presented = 0 # 全部诊断的case

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    max_cases = min(num_scenarios, scenario_loader.num_scenarios) # 诊断的case量
    
    evaluator_agent = DiagnosisEvaluator() # 评价 agent 要全程记录，所有提前初始化


    doctor_llm = AgentsLLM(
        os.getenv("MODEL_ID"),  # 暂时都用同一个model
        os.getenv("API_KEY"),
        os.getenv("BASE_URL"),
    )

    patient_llm = AgentsLLM(
        os.getenv("MODEL_ID"),
        os.getenv("API_KEY"),
        os.getenv("BASE_URL"),
    )

    intake_llm = AgentsLLM(
        os.getenv("MODEL_ID"),
        os.getenv("API_KEY"),
        os.getenv("BASE_URL"),
    )

    for scenario_id in range(max_cases):
        total_presented += 1
        scenario = scenario_loader.get_scenario(id=scenario_id)

        patient_agent = agents.PatientAgent(
            scenario=scenario,
            llm_client=patient_llm,
        )

        doctor_agent = agents.DoctorAgent(
            scenario=scenario,
            llm_client=doctor_llm,
            max_infs=max_infs,
            img_processing=img_processing,
        )

        intake_agent = agents.IntakeAgent(
            scenario=scenario,
            llm_client=intake_llm,  # intake agent 和 doctor agent 共用一个 llm client
            max_infs=max_infs,
            img_processing=img_processing,
        )

        measurement_agent = MeasurementAgent(scenario=scenario)

        case_store = CaseStore(scenario_id) # 用于 intake agent 记录病历信息，最终输出结构化病历

        # ======== 一次会诊模拟开始 ========
        current_observation = "Please begin the consultation. Ask the patient your first question." # begin conversation
        
        print(f"\n{'=' * 80}")
        print(f"Scenario {scenario_id}")
        print(f"Examiner info: {scenario.examiner_info}")
        print(f"{'=' * 80}\n")

        
        for inf_id in range(max_infs):
            progress = int(((inf_id + 1) / max_infs) * 100)

            # ========== Doctor ==========
            if inf_id == max_infs - 1:
                current_observation += (
                    "\nThis is the final turn. If you have enough evidence, provide your diagnosis."
                )

            doctor_dialogue = _doctor_say(inf_type, doctor_agent, current_observation)

            if doctor_dialogue is None:
                print("DoctorAgent returned None, skipping this case.")
                break

            print(f"Doctor [{progress}%]: {doctor_dialogue}")
            time.sleep(1.0)

 

            

            # ---------- Doctor 输出最终诊断 ----------  
            if is_diagnosis_ready(doctor_dialogue):
                doctor_final = extract_final_diagnosis(doctor_dialogue)
                correct_answer = scenario.diagnosis_info

                judge = evaluator_agent.evaluate(
                    diagnosis=doctor_final,
                    correct_diagnosis=correct_answer,
                )

                correctness = judge.lower() == "yes"
                if correctness:
                    total_correct += 1

                accuracy = int((total_correct / total_presented) * 100)

                print("\nFinal diagnosis:", doctor_final)
                print("Correct answer:", correct_answer)
                print(
                    f"Scene {scenario_id}, The diagnosis was",
                    "CORRECT" if correctness else "INCORRECT",
                    accuracy,
                )
                break

            # ---------- Doctor 请求图像 ----------
            if is_request_images(doctor_dialogue):
                if hasattr(scenario, "image_url"):
                    current_observation = "The requested medical image is now available. Please continue your reasoning."
                else:
                    current_observation = "No medical image is available for this case. Please continue without images."

                doctor_agent.add_hist(current_observation)
                time.sleep(1.0)
                continue

            # ---------- Doctor 请求检查 ----------
            if is_request_test(doctor_dialogue):
                measurement_dialogue = measurement_agent.inference_measurement(doctor_dialogue)

                print(f"Measurement [{progress}%]: {measurement_dialogue}")

                if measurement_dialogue is None:
                    print("MeasurementAgent returned None, skipping this case.")
                    break

                doctor_agent.add_hist(measurement_dialogue)
                current_observation = measurement_dialogue  # ✅ 补上（更合理）

                time.sleep(1.0)
                continue

            # ---------- Patient 回复 Doctor ----------
            patient_dialogue = _patient_say(inf_type, patient_agent, doctor_dialogue)

            if patient_dialogue is None:
                print("PatientAgent returned None, skipping this case.")
                break

            print(f"Patient [{progress}%]: {patient_dialogue}")

            current_observation = patient_dialogue
            time.sleep(1.0)

    print(f"\nFinished {total_presented} scenarios.")
    print(f"Total correct: {total_correct}")

    if total_presented > 0:
        print(f"Accuracy: {100 * total_correct / total_presented:.2f}%")
    else:
        print("Accuracy: N/A")






# ==============================================================

def _doctor_say(inf_type, doctor_agent, current_observation):
    if inf_type == "human_doctor":
        doctor_dialogue = input("\nDoctor: ") # human input
    else:
        doctor_dialogue = doctor_agent.inference(current_observation)

        if doctor_dialogue is not None:
            doctor_agent.add_hist(
                f"Patient: {current_observation}\nDoctor: {doctor_dialogue}"
            )
    return doctor_dialogue


def _patient_say(inf_type, patient_agent, doctor_dialogue):
    if inf_type == "human_patient":
        patient_dialogue = input("\nPatient: ")
    else:
        patient_dialogue = patient_agent.inference(doctor_dialogue)

        if patient_dialogue is not None:
            patient_agent.add_hist(
                f"Doctor: {doctor_dialogue}\nPatient: {patient_dialogue}"
            )

    return patient_dialogue




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Medical Diagnosis Simulation System") 

    # 创建一个命令行参数解析器 
    # eg: 
    #   python main.py --inf_type human_doctor --agent_dataset NEJM --num_scenarios 5 --max_infs 10 --img_processing

    parser.add_argument("--inf_type", type=str, choices=["llm", "human_doctor", "human_patient"], default="llm", 
                        help="Type of inference: 'llm' for fully automated, 'human_doctor' to manually control the doctor, 'human_patient' to manually control the patient.")
    parser.add_argument("--agent_dataset", type=str, default="MedQA", choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext", "MIMICIV"], help="Dataset to use for simulation.")
    parser.add_argument("--img_processing", action="store_true", help="Enable the doctor to request medical images (if available in the dataset).")
    parser.add_argument("--num_scenarios", type=int, default=None, help="Number of clinical cases to run. Defaults to all available cases.")
    parser.add_argument("--max_infs", type=int, default=20, help="Maximum number of interaction turns between doctor and patient.")

    args = parser.parse_args()

    main(
        inf_type=args.inf_type,
        dataset=args.agent_dataset,
        num_scenarios=args.num_scenarios,
        max_infs=args.max_infs,
        img_processing=args.img_processing,
    )



