import json
import torch
import platform
import datetime
import socket
import uuid
import os
from pathlib import Path
from typing import Dict, Any, Union, List

from config import config


class CheckpointManager:
    """
    CycloneDX v1.5 표준을 준수하는 AI BOM 생성 및 모델 체크포인트 매니저
    """

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
    def _create_cyclonedx_bom(
        model_name: str,
        model_spec: Dict[str, Any],
        train_config: Dict[str, Any],
        extra_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        CycloneDX v1.5 JSON 스키마에 맞춰 데이터 구조 생성
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bom_uuid = str(uuid.uuid4())

        # 1. 학습 하이퍼파라미터 (modelParameters)
        hyperparameters = []
        for k, v in train_config.items():
            hyperparameters.append({"name": k, "value": str(v)})

        # 2. 성능 지표 (quantitativeAnalysis)
        performance_metrics = []
        if extra_metadata:
            for k, v in extra_metadata.items():
                performance_metrics.append({"type": k, "value": str(v)})

        # 3. 메인 컴포넌트 (Machine Learning Model)
        component = {
            "type": "machine-learning-model",
            "name": model_name,
            "version": "1.0.0",
            "bom-ref": f"model-{bom_uuid}",
            "description": f"Mafia AI Agent ({model_spec.get('algorithm', 'unknown')})",
            "properties": CheckpointManager.get_system_properties(),
            "modelCard": {
                "modelParameters": {
                    "approach": {
                        "type": (
                            "supervised"
                            if "il" in str(train_config).lower()
                            else "reinforcement"
                        )
                    },
                    "task": "game-playing",
                    "architectureFamily": model_spec.get("backbone", "mlp"),
                    "modelArchitecture": f"Hidden: {model_spec.get('structure', {}).get('hidden_dim')}, Layers: {model_spec.get('structure', {}).get('num_layers')}",
                    "datasets": [
                        {"name": "Self-Play Simulation Logs", "type": "synthetic"}
                    ],
                },
                "quantitativeAnalysis": {"performanceMetrics": performance_metrics},
            },
        }

        # 하이퍼파라미터를 properties에도 추가 (검색 편의성)
        for hp in hyperparameters:
            component["properties"].append(
                {"name": f"hyperparameter:{hp['name']}", "value": hp["value"]}
            )

        # 4. 최종 BOM 구조 조립
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
                    "version": "1.0",
                },
                "component": {
                    "type": "application",
                    "name": "Mafia AI Training System",
                    "version": "1.0",
                },
            },
            "components": [component],
        }

        return bom, bom_uuid

    @staticmethod
    def save_checkpoint(
        filepath: Union[str, Path],
        state_dict: Dict[str, Any],
        model_spec: Dict[str, Any],
        extra_metadata: Dict[str, Any] = None,
    ):
        """
        모델(.pt)과 CycloneDX 표준 BOM(.json) 저장
        """
        path_obj = Path(filepath)
        base_name = path_obj.stem
        parent_dir = path_obj.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Config 가져오기
        train_config = {}
        if hasattr(config, "train"):
            train_config = (
                config.train.model_dump()
                if hasattr(config.train, "model_dump")
                else config.train.dict()
            )

        # CycloneDX BOM 생성
        bom_data, bom_uuid = CheckpointManager._create_cyclonedx_bom(
            model_name=base_name,
            model_spec=model_spec,
            train_config=train_config,
            extra_metadata=extra_metadata,
        )

        # JSON 파일 저장 (.cdx.json 확장자 권장)
        json_path = parent_dir / f"{base_name}.cdx.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bom_data, f, indent=2)

        # PT 파일 저장 (BOM 연결 정보 포함)
        state_dict["bom_uuid"] = bom_uuid
        state_dict["bom_format"] = "CycloneDX_1.5"
        state_dict["bom_path"] = str(json_path.name)

        # 모델 복원용 필수 데이터 병합 (안전장치)
        for key, value in model_spec.items():
            if key != "structure":
                state_dict[key] = value
        if "structure" in model_spec:
            state_dict.update(model_spec["structure"])

        torch.save(state_dict, filepath)

        print(f"[Checkpoint] Model saved: {filepath}")
        print(f"[Checkpoint] BOM saved:   {json_path}")

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
