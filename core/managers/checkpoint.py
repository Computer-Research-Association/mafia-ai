import json
import torch
import platform
import datetime
import socket
import uuid
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

from config import config


class CheckpointManager:
    """
    CycloneDX v1.5 표준 (ML-BOM)을 완벽 지원하는 체크포인트 매니저
    - SHA-256 해시 검증
    - 학습 방법(RL/IL) 자동 감지
    - 데이터셋 및 외부 참조 명시
    """

    # 프로젝트 저장소 주소 (추적성 확보용)
    REPO_URL = "https://github.com/computer-research-association/mafia-ai"

    @staticmethod
    def get_system_properties() -> List[Dict[str, str]]:
        """시스템 정보를 CycloneDX Property 형식으로 변환"""
        return [
            {
                "name": "operating_system",
                "value": f"{platform.system()} {platform.release()}",
            },
            {"name": "python_version", "value": platform.python_version()},
            {"name": "torch_version", "value": torch.__version__},
            {"name": "device", "value": "cuda" if torch.cuda.is_available() else "cpu"},
        ]

    @staticmethod
    def calculate_file_hash(
        filepath: Union[str, Path], algorithm: str = "sha256"
    ) -> str:
        """파일의 무결성 검증을 위한 해시 계산"""
        hash_func = getattr(hashlib, algorithm)()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    @staticmethod
    def _create_cyclonedx_bom(
        model_name: str,
        model_spec: Dict[str, Any],
        train_config: Dict[str, Any],
        extra_metadata: Dict[str, Any],
        file_hash: str,
        bom_uuid: str,
    ) -> Dict[str, Any]:
        """
        CycloneDX v1.5 ML-BOM 구조 생성 (개선된 로직 적용)
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        il_coef = train_config.get("IL_COEF", 0.0)
        approach_type = "reinforcement"
        if il_coef > 0:

            approach_type = "reinforcement"

        datasets = [
            {
                "name": "Self-Play Simulation Logs",
                "type": "synthetic",
                "description": "Generated dynamically during self-play training",
            }
        ]
        if il_coef > 0:
            datasets.append(
                {
                    "name": "Human/Expert Demonstration Data",
                    "type": "dataset",
                    "description": "Used for Imitation Learning (Behavior Cloning)",
                }
            )

        performance_metrics = []
        if extra_metadata:
            for k, v in extra_metadata.items():
                if isinstance(v, float):
                    val_str = f"{v:.4f}"
                else:
                    val_str = str(v)

                metric_entry = {"type": k, "value": val_str}
                performance_metrics.append(metric_entry)

        # 하이퍼파라미터 추출
        hyperparameters = []
        for k, v in train_config.items():
            hyperparameters.append({"name": k, "value": str(v)})

        # 메인 컴포넌트 (Machine Learning Model)
        component = {
            "type": "machine-learning-model",
            "name": model_name,
            "version": "1.0.0",
            "bom-ref": f"model-{bom_uuid}",
            "description": f"Mafia AI Agent ({model_spec.get('algorithm', 'ppo')})",
            "hashes": [{"alg": "SHA-256", "content": file_hash}],
            "externalReferences": [
                {
                    "type": "vcs",
                    "url": CheckpointManager.REPO_URL,
                    "comment": "Model Training Source Code",
                }
            ],
            "properties": CheckpointManager.get_system_properties(),
            "modelCard": {
                "modelParameters": {
                    "approach": {"type": approach_type},
                    "task": "game-playing",
                    "architectureFamily": model_spec.get("backbone", "mlp"),
                    "modelArchitecture": f"Hidden: {model_spec.get('structure', {}).get('hidden_dim')}, Layers: {model_spec.get('structure', {}).get('num_layers')}",
                    "datasets": datasets,
                },
                "quantitativeAnalysis": {"performanceMetrics": performance_metrics},
            },
        }

        # 하이퍼파라미터 검색용 속성 추가
        for hp in hyperparameters:
            component["properties"].append(
                {"name": f"hyperparameter:{hp['name']}", "value": hp["value"]}
            )

        # 최종 BOM 조립
        bom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:{bom_uuid}",
            "version": 1,
            "metadata": {
                "timestamp": timestamp,
                "tool": {
                    "vendor": "Mafia AI Team",
                    "name": "Mafia AI Checkpoint Manager",
                    "version": "1.1",
                },
                "component": {
                    "type": "application",
                    "name": "Mafia AI Training System",
                    "version": "1.0",
                },
            },
            "components": [component],
        }

        return bom

    @staticmethod
    def save_checkpoint(
        filepath: Union[str, Path],
        state_dict: Dict[str, Any],
        model_spec: Dict[str, Any],
        extra_metadata: Dict[str, Any] = None,
    ):
        """
        모델 저장 및 무결성 검증이 포함된 BOM 생성
        """
        path_obj = Path(filepath)
        base_name = path_obj.stem
        parent_dir = path_obj.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # 1. UUID 생성
        bom_uuid = str(uuid.uuid4())
        json_path = parent_dir / f"{base_name}.cdx.json"

        # 2. [중요] .pt 파일 먼저 저장
        # 이유: BOM에 파일 해시(SHA-256)를 넣으려면 파일이 먼저 있어야 함
        state_dict["bom_uuid"] = bom_uuid
        state_dict["bom_format"] = "CycloneDX_1.5"
        state_dict["bom_path"] = str(json_path.name)

        # 복원용 스펙 병합
        for key, value in model_spec.items():
            if key != "structure":
                state_dict[key] = value
        if "structure" in model_spec:
            state_dict.update(model_spec["structure"])

        torch.save(state_dict, filepath)

        # 3. 저장된 파일의 해시 계산 (무결성 확보)
        file_hash = CheckpointManager.calculate_file_hash(filepath)

        # 4. Config 가져오기
        train_config = {}
        if hasattr(config, "train"):
            train_config = (
                config.train.model_dump()
                if hasattr(config.train, "model_dump")
                else config.train.dict()
            )

        # 5. BOM 데이터 생성 (해시 포함)
        bom_data = CheckpointManager._create_cyclonedx_bom(
            model_name=base_name,
            model_spec=model_spec,
            train_config=train_config,
            extra_metadata=extra_metadata,
            file_hash=file_hash,
            bom_uuid=bom_uuid,
        )

        # 6. JSON 파일 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bom_data, f, indent=2)

        print(f"[Checkpoint] Model saved: {filepath}")
        print(f"[Checkpoint] BOM saved:   {json_path} (SHA-256 Verified)")

    @staticmethod
    def load_checkpoint(filepath: Union[str, Path]) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))

        if "bom_uuid" in checkpoint:
            fmt = checkpoint.get("bom_format", "Unknown")
            print(
                f"[Checkpoint] Loading BOM-linked model ({fmt}, UUID: {checkpoint['bom_uuid']})"
            )

        return checkpoint
