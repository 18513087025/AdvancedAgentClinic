from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import os
    



class CaseStore:
    """
    整个病例在系统中的共享状态仓库。

    设计原则：
    1. 只存“共享状态”，不存各个 agent 的本地 history
    2. patient_info 作为病人事实区
    3. specialist_opinions / triage / final_report 作为推理结果区
    4. 支持 patch 式更新，避免后写覆盖前写
    """

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.data: dict[str, Any] = {
            "case_id": case_id,

            # 共享结构化事实
            "patient_info": {

                # 在诊断开始时全部提供还是渐进式披露

                "patient_profile": {},   # IntakeAgent 基于完整对话历史生成
                "physical_exam": {},     # 客观检查所见 
                "labs": {},              # 实验室检查结果
                "imaging": {},           # 影像检查结果
                "pathology": {},         # 病理报告 / 切片结果
            },
            
            "final_report":""

        }

    # =========================
    # 读接口
    # =========================
    def get_view(self, patient_info_empowered: bool = False) -> dict[str, Any]:
        view: dict[str, Any] = {"case_id": self.case_id}

        if patient_info_empowered:
            view["patient_info"] = deepcopy(self.data["patient_info"])

        return view



    # =========================
    # 写接口
    # =========================


    def set_patient_profile(self, patient_profile: dict[str, Any]) -> None:
        self.data["patient_info"]["patient_profile"] = deepcopy(patient_profile)

    def update_physical_exam(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["physical_exam"], patch)

    def update_labs(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["labs"], patch)

    def update_imaging(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["imaging"], patch)

    def update_pathology(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["pathology"], patch)

    # =========================
    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.data)

    def save_json(self, path: str, append: bool = True):
        """
        Save case to file.

        Args:
            path: 文件路径
            append: True=追加写入, False=覆盖文件
        """

        mode = "a" if append else "w"

        record = self.to_dict()  # 深拷贝安全
        record["case_id"] = self.case_id  # 确保有 id
        os.makedirs(os.path.dirname(path), exist_ok=True)  
        with open(path, mode, encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _deep_merge(self, target: dict[str, Any], patch: dict[str, Any]) -> None:
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                self._deep_merge(target[k], v)
            else:
                target[k] = deepcopy(v)


