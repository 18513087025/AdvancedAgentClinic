import argparse
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from agents.doctor_agent import DoctorAgent
from agents.patient_agent import PatientAgent
from agents.evaluator_agent import DiagnosisEvaluator
from agents.measurement_agent import MeasurementAgent

from data.scenario_loader import (
    ScenarioLoaderMedQA,
    ScenarioLoaderMedQAExtended,
    ScenarioLoaderMIMICIV,
    ScenarioLoaderNEJM,
    ScenarioLoaderNEJMExtended,
)

from utils.parsing import (
    is_diagnosis_ready,
    is_request_test,
    is_request_images,
    extract_final_diagnosis,
)


# =========================
# Dataset selector
# =========================
def get_scenario_loader(dataset: str):
    """Return the corresponding scenario loader based on dataset name."""
    if dataset == "MedQA":
        return ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        return ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        return ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        return ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        return ScenarioLoaderMIMICIV()
    else:
        raise ValueError(f"Dataset {dataset} does not exist")


# =========================
# Main loop
# =========================
def main(
    inf_type: str = "llm",
    dataset: str = "MedQA",
    num_scenarios: Optional[int] = None,
    total_inferences: int = 20,
    doctor_image_request: bool = False,
):
    scenario_loader = get_scenario_loader(dataset)
    evaluator = DiagnosisEvaluator()

    total_correct = 0
    total_presented = 0

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    max_cases = min(num_scenarios, scenario_loader.num_scenarios)

    for scenario_id in range(max_cases):
        total_presented += 1
        scenario = scenario_loader.get_scenario(id=scenario_id)

        patient_agent = PatientAgent(scenario=scenario)
        doctor_agent = DoctorAgent(
            scenario=scenario,
            max_infs=total_inferences,
            img_request=doctor_image_request,
        )
        measurement_agent = MeasurementAgent(scenario=scenario)

        current_observation = "Please begin the consultation. Ask the patient your first question."
        image_requested = False

        print(f"\n{'=' * 80}")
        print(f"Scenario {scenario_id}")
        print(f"Examiner info: {scenario.examiner_information()}")
        print(f"{'=' * 80}\n")

        for inf_id in range(total_inferences):
            if inf_id == total_inferences - 1:
                current_observation += (
                    "\nThis is the final turn. If you have enough evidence, provide your diagnosis."
                )

            # ---------- Doctor ----------
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nDoctor: ")
            else:
                scene = scenario if hasattr(scenario, "image_url") else None

                doctor_dialogue = doctor_agent.inference_doctor(
                    patient_response=current_observation,
                    scene=scene,
                    image_requested=image_requested,
                )

            print(f"Doctor [{int(((inf_id + 1) / total_inferences) * 100)}%]: {doctor_dialogue}")

            if doctor_dialogue is None:
                print("DoctorAgent returned None, skipping this case.")
                break

            # ---------- Final Diagnosis ----------
            if is_diagnosis_ready(doctor_dialogue):
                doctor_final = extract_final_diagnosis(doctor_dialogue)
                correct_answer = scenario.diagnosis_information()

                judge = evaluator.evaluate(
                    diagnosis=doctor_final,
                    correct_diagnosis=correct_answer,
                )

                correctness = judge == "yes"
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

            # ---------- Request Images ----------
            if is_request_images(doctor_dialogue):
                if dataset in {"NEJM", "NEJM_Ext"} and hasattr(scenario, "image_url"):
                    current_observation = (
                        "The requested medical image is now available. Please continue your reasoning."
                    )
                    image_requested = True
                else:
                    current_observation = (
                        "No medical image is available for this case. Please continue without images."
                    )
                    image_requested = False

                time.sleep(1.0)
                continue

            # ---------- Request Test ----------
            if is_request_test(doctor_dialogue):
                measurement_dialogue = measurement_agent.inference_measurement(doctor_dialogue)

                print(
                    f"Measurement [{int(((inf_id + 1) / total_inferences) * 100)}%]: {measurement_dialogue}"
                )

                if measurement_dialogue is None:
                    print("MeasurementAgent returned None, skipping this case.")
                    break

                current_observation = measurement_dialogue

                patient_agent.add_hist(measurement_dialogue)
                doctor_agent.add_hist(measurement_dialogue)

                image_requested = False
                continue

            # ---------- Patient ----------
            if inf_type == "human_patient":
                patient_dialogue = input("\nPatient: ")
            else:
                patient_dialogue = patient_agent.inference_patient(doctor_dialogue)

            print(f"Patient [{int(((inf_id + 1) / total_inferences) * 100)}%]: {patient_dialogue}")

            if patient_dialogue is None:
                print("PatientAgent returned None, skipping this case.")
                break

            current_observation = patient_dialogue
            measurement_agent.add_hist(patient_dialogue)
            image_requested = False

            time.sleep(1.0)

    print(f"\nFinished {total_presented} scenarios.")
    print(f"Total correct: {total_correct}")

    if total_presented > 0:
        print(f"Accuracy: {100 * total_correct / total_presented:.2f}%")
    else:
        print("Accuracy: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent Medical Diagnosis Simulation System"
    )

    parser.add_argument(
        "--inf_type",
        type=str,
        choices=["llm", "human_doctor", "human_patient"],
        default="llm",
        help="Type of inference: 'llm' for fully automated, "
             "'human_doctor' to manually control the doctor, "
             "'human_patient' to manually control the patient."
    )

    parser.add_argument(
        "--agent_dataset",
        type=str,
        default="MedQA",
        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext", "MIMICIV"],
        help="Dataset to use for simulation."
    )

    parser.add_argument(
        "--doctor_image_request",
        action="store_true",
        help="Enable the doctor to request medical images (if available in the dataset)."
    )

    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=None,
        help="Number of clinical cases to run. Defaults to all available cases."
    )

    parser.add_argument(
        "--total_inferences",
        type=int,
        default=20,
        help="Maximum number of interaction turns between doctor and patient."
    )

    args = parser.parse_args()

    main(
        inf_type=args.inf_type,
        dataset=args.agent_dataset,
        num_scenarios=args.num_scenarios,
        total_inferences=args.total_inferences,
        doctor_image_request=args.doctor_image_request,
    )