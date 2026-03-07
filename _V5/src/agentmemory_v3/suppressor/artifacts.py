from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig
from agentmemory_v3.utils.io import read_jsonl

from .model import OreoMemoryMLP, OreoTypeHeadsMLP, PlainSuppressorMLP


@dataclass(frozen=True)
class SuppressorManifest:
    version: str
    model_type: str
    encoder_model_name: str
    use_e5_prefix: bool
    local_files_only: bool
    offline: bool
    device: str
    batch_size: int
    embedding_dim: int
    feature_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float
    include_query_vec: bool
    include_memory_vec: bool
    include_cosine: bool
    include_product: bool
    include_abs_diff: bool
    include_feedback_type: bool
    include_lane: bool = False
    include_lexical_overlap: bool = True
    include_explicit_mention: bool = True
    objective: str = "pointwise"
    slots_k: int = 0
    top_r: int = 0
    tau: float = 1.0
    feedback_types: tuple[str, ...] = ()
    enable_memory_bias: bool = False
    lambda_memory_bias: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict) -> "SuppressorManifest":
        return cls(
            version=str(payload.get("version") or "plain_mlp_v1"),
            model_type=str(payload.get("model_type") or "plain_mlp_v1"),
            encoder_model_name=str(payload.get("encoder_model_name") or "intfloat/multilingual-e5-small"),
            use_e5_prefix=bool(payload.get("use_e5_prefix", True)),
            local_files_only=bool(payload.get("local_files_only", True)),
            offline=bool(payload.get("offline", True)),
            device=str(payload.get("device") or "auto"),
            batch_size=int(payload.get("batch_size") or 128),
            embedding_dim=int(payload.get("embedding_dim") or 384),
            feature_dim=int(payload.get("feature_dim") or 0),
            hidden_dims=tuple(int(item) for item in (payload.get("hidden_dims") or [512, 128])),
            dropout=float(payload.get("dropout") or 0.15),
            include_query_vec=bool(payload.get("include_query_vec", True)),
            include_memory_vec=bool(payload.get("include_memory_vec", True)),
            include_cosine=bool(payload.get("include_cosine", True)),
            include_product=bool(payload.get("include_product", True)),
            include_abs_diff=bool(payload.get("include_abs_diff", True)),
            include_feedback_type=bool(payload.get("include_feedback_type", True)),
            include_lane=bool(payload.get("include_lane", False)),
            include_lexical_overlap=bool(payload.get("include_lexical_overlap", True)),
            include_explicit_mention=bool(payload.get("include_explicit_mention", True)),
            objective=str(payload.get("objective") or "pointwise"),
            slots_k=int(payload.get("slots_k") or 0),
            top_r=int(payload.get("top_r") or 0),
            tau=float(payload.get("tau") or 1.0),
            feedback_types=tuple(str(item) for item in (payload.get("feedback_types") or [])),
            enable_memory_bias=bool(payload.get("enable_memory_bias", False)),
            lambda_memory_bias=float(payload.get("lambda_memory_bias") or 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "encoder_model_name": self.encoder_model_name,
            "use_e5_prefix": self.use_e5_prefix,
            "local_files_only": self.local_files_only,
            "offline": self.offline,
            "device": self.device,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "include_query_vec": self.include_query_vec,
            "include_memory_vec": self.include_memory_vec,
            "include_cosine": self.include_cosine,
            "include_product": self.include_product,
            "include_abs_diff": self.include_abs_diff,
            "include_feedback_type": self.include_feedback_type,
            "include_lane": self.include_lane,
            "include_lexical_overlap": self.include_lexical_overlap,
            "include_explicit_mention": self.include_explicit_mention,
            "objective": self.objective,
            "slots_k": self.slots_k,
            "top_r": self.top_r,
            "tau": self.tau,
            "feedback_types": list(self.feedback_types),
            "enable_memory_bias": self.enable_memory_bias,
            "lambda_memory_bias": self.lambda_memory_bias,
        }


@dataclass
class SuppressorArtifacts:
    artifact_dir: Path
    manifest: SuppressorManifest
    model: torch.nn.Module
    encoder: HFSentenceEncoder
    memory_ids: list[str]
    memory_matrix: np.ndarray
    memory_text_by_id: dict[str, str]
    memory_index_by_id: dict[str, int]
    memory_bias_by_id: dict[str, float]
    calibration_payload: dict

    @staticmethod
    def _build_model(manifest: SuppressorManifest) -> torch.nn.Module:
        model_type = str(manifest.model_type or "").strip().lower()
        if model_type == "oreo_type_heads_mlp_v1":
            return OreoTypeHeadsMLP(
                input_dim=int(manifest.feature_dim),
                hidden_dims=manifest.hidden_dims,
                dropout=float(manifest.dropout),
                slots_k=max(1, int(manifest.slots_k or 32)),
                top_r=max(1, int(manifest.top_r or 2)),
                tau=max(1e-4, float(manifest.tau or 0.7)),
                feedback_types=manifest.feedback_types or ("unrelated", "toforget"),
            )
        if model_type == "oreo_memory_mlp_v1":
            return OreoMemoryMLP(
                input_dim=int(manifest.feature_dim),
                hidden_dims=manifest.hidden_dims,
                dropout=float(manifest.dropout),
                slots_k=max(1, int(manifest.slots_k or 32)),
                top_r=max(1, int(manifest.top_r or 2)),
                tau=max(1e-4, float(manifest.tau or 0.7)),
            )
        return PlainSuppressorMLP(
            input_dim=int(manifest.feature_dim),
            hidden_dims=manifest.hidden_dims,
            dropout=float(manifest.dropout),
        )

    @staticmethod
    def _default_calibration() -> dict:
        return {
            "version": "temperature_bias_v1",
            "default": {"temperature": 1.0, "bias": 0.0},
            "by_type": {},
            "by_lane_type": {},
        }

    def get_calibration_params(self, *, lane: str, feedback_type: str) -> tuple[float, float, str]:
        payload = self.calibration_payload or {}
        lane_key = str(lane or "").strip().lower()
        type_key = str(feedback_type or "").strip().lower()

        by_lane_type = payload.get("by_lane_type") or {}
        lane_map = by_lane_type.get(lane_key) or {}
        item = lane_map.get(type_key)
        if isinstance(item, dict):
            return (
                max(1e-3, float(item.get("temperature", 1.0))),
                float(item.get("bias", 0.0)),
                "lane_type",
            )

        by_type = payload.get("by_type") or {}
        item = by_type.get(type_key)
        if isinstance(item, dict):
            return (
                max(1e-3, float(item.get("temperature", 1.0))),
                float(item.get("bias", 0.0)),
                "type",
            )

        default_item = payload.get("default") or {}
        return (
            max(1e-3, float(default_item.get("temperature", 1.0))),
            float(default_item.get("bias", 0.0)),
            "default",
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "SuppressorArtifacts":
        base = Path(artifact_dir)
        manifest = SuppressorManifest.from_dict(json.loads((base / "manifest.json").read_text(encoding="utf-8")))
        model = cls._build_model(manifest)
        state = torch.load(base / "model.pt", map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        encoder = HFSentenceEncoder(
            SentenceEncoderConfig(
                model_name=manifest.encoder_model_name,
                use_e5_prefix=manifest.use_e5_prefix,
                local_files_only=manifest.local_files_only,
                offline=manifest.offline,
                device=manifest.device,
                batch_size=manifest.batch_size,
            )
        )
        memory_ids = [str(item) for item in json.loads((base / "memory_ids.json").read_text(encoding="utf-8"))]
        memory_matrix = np.load(base / "memory_embeddings.npy").astype(np.float32)
        memory_text_by_id: dict[str, str] = {}
        text_path = base / "memory_texts.jsonl"
        if text_path.exists():
            for row in read_jsonl(text_path):
                memory_id = str(row.get("memory_id") or "")
                if memory_id:
                    memory_text_by_id[memory_id] = str(row.get("memory_text") or row.get("display_text") or "")
        memory_bias_by_id: dict[str, float] = {}
        bias_path = base / "memory_bias.json"
        if bias_path.exists():
            try:
                bias_payload = json.loads(bias_path.read_text(encoding="utf-8"))
                for memory_id, value in (bias_payload.get("bias_by_id") or {}).items():
                    memory_bias_by_id[str(memory_id)] = float(value)
            except Exception:
                memory_bias_by_id = {}
        calibration_payload = cls._default_calibration()
        calibration_path = base / "calibration.json"
        if calibration_path.exists():
            try:
                loaded = json.loads(calibration_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    calibration_payload = {
                        "version": str(loaded.get("version") or "temperature_bias_v1"),
                        "default": dict(loaded.get("default") or {"temperature": 1.0, "bias": 0.0}),
                        "by_type": dict(loaded.get("by_type") or {}),
                        "by_lane_type": dict(loaded.get("by_lane_type") or {}),
                        "stats": loaded.get("stats") or {},
                    }
            except Exception:
                calibration_payload = cls._default_calibration()
        memory_index_by_id = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
        return cls(
            artifact_dir=base,
            manifest=manifest,
            model=model,
            encoder=encoder,
            memory_ids=memory_ids,
            memory_matrix=memory_matrix,
            memory_text_by_id=memory_text_by_id,
            memory_index_by_id=memory_index_by_id,
            memory_bias_by_id=memory_bias_by_id,
            calibration_payload=calibration_payload,
        )
