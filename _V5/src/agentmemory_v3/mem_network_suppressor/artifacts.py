from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig
from agentmemory_v3.utils.io import read_jsonl

from .features import (
    FEEDBACK_TYPE_VOCAB_SIZE,
    LANE_VOCAB_SIZE,
    UNKNOWN_MEMORY_ID_INDEX,
    build_memory_id_lookup,
    build_feedback_type_lane_arrays,
    memory_id_to_index,
)
from .model import LegacyMLPScorer, MemNetworkScorer


@dataclass(frozen=True)
class MemNetworkManifest:
    version: str = "mem_network_suppressor_v3"
    model_type: str = "mem_network_single_head_v3"
    encoder_model_name: str = "intfloat/multilingual-e5-small"
    use_e5_prefix: bool = True
    local_files_only: bool = True
    offline: bool = True
    device: str = "auto"
    batch_size: int = 128
    embedding_dim: int = 384
    # Legacy-only field kept for compatibility.
    feature_dim: int = 1152
    hidden_dims: tuple[int, ...] = (640, 192)
    state_dim: int = 256
    num_hops: int = 1
    num_heads: int = 1
    memory_id_vocab_size: int = 2
    use_memory_id_addressing: bool = True
    type_vocab_size: int = FEEDBACK_TYPE_VOCAB_SIZE
    lane_vocab_size: int = LANE_VOCAB_SIZE
    type_emb_dim: int = 16
    lane_emb_dim: int = 8
    dropout: float = 0.15
    feedback_top_k: int = 8
    suppress_strength: float = 0.35

    @classmethod
    def from_dict(cls, payload: dict) -> "MemNetworkManifest":
        model_type = str(payload.get("model_type") or "mem_network_single_head_v3")
        default_num_heads = 2 if "dual" in model_type.lower() else 1
        return cls(
            version=str(payload.get("version") or "mem_network_suppressor_v3"),
            model_type=model_type,
            encoder_model_name=str(payload.get("encoder_model_name") or "intfloat/multilingual-e5-small"),
            use_e5_prefix=bool(payload.get("use_e5_prefix", True)),
            local_files_only=bool(payload.get("local_files_only", True)),
            offline=bool(payload.get("offline", True)),
            device=str(payload.get("device") or "auto"),
            batch_size=int(payload.get("batch_size") or 128),
            embedding_dim=int(payload.get("embedding_dim") or 384),
            feature_dim=int(payload.get("feature_dim") or 1152),
            hidden_dims=tuple(int(item) for item in (payload.get("hidden_dims") or [640, 192])),
            state_dim=max(32, int(payload.get("state_dim") or 256)),
            num_hops=max(1, int(payload.get("num_hops") or 1)),
            num_heads=max(1, int(payload.get("num_heads") or default_num_heads)),
            memory_id_vocab_size=max(2, int(payload.get("memory_id_vocab_size") or 2)),
            use_memory_id_addressing=bool(payload.get("use_memory_id_addressing", True)),
            type_vocab_size=max(2, int(payload.get("type_vocab_size") or FEEDBACK_TYPE_VOCAB_SIZE)),
            lane_vocab_size=max(3, int(payload.get("lane_vocab_size") or LANE_VOCAB_SIZE)),
            type_emb_dim=max(4, int(payload.get("type_emb_dim") or 16)),
            lane_emb_dim=max(4, int(payload.get("lane_emb_dim") or 8)),
            dropout=float(payload.get("dropout") or 0.15),
            feedback_top_k=max(1, int(payload.get("feedback_top_k") or 8)),
            suppress_strength=float(payload.get("suppress_strength") or 0.35),
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
            "state_dim": self.state_dim,
            "num_hops": self.num_hops,
            "num_heads": self.num_heads,
            "memory_id_vocab_size": self.memory_id_vocab_size,
            "use_memory_id_addressing": self.use_memory_id_addressing,
            "type_vocab_size": self.type_vocab_size,
            "lane_vocab_size": self.lane_vocab_size,
            "type_emb_dim": self.type_emb_dim,
            "lane_emb_dim": self.lane_emb_dim,
            "dropout": self.dropout,
            "feedback_top_k": self.feedback_top_k,
            "suppress_strength": self.suppress_strength,
        }


@dataclass
class MemNetworkArtifacts:
    artifact_dir: Path
    manifest: MemNetworkManifest
    model: LegacyMLPScorer | MemNetworkScorer
    is_legacy_model: bool
    encoder: HFSentenceEncoder
    memory_ids: list[str]
    memory_matrix: np.ndarray
    memory_index_by_id: dict[str, int]
    memory_text_by_id: dict[str, str]
    feedback_rows: list[dict]
    feedback_query_matrix: np.ndarray
    feedback_memory_matrix: np.ndarray
    feedback_memory_id_ids: np.ndarray
    feedback_type_ids: np.ndarray
    feedback_lane_ids: np.ndarray
    suppress_memory_ids: list[str]
    suppress_memory_id_to_index: dict[str, int]

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "MemNetworkArtifacts":
        base = Path(artifact_dir)
        manifest_path = base / "manifest.json"
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest = MemNetworkManifest.from_dict(manifest_payload)

        is_legacy_model = str(manifest.model_type).strip().lower() in {"mem_network_mlp_v1", "mem_network_scorer_v1"}
        if is_legacy_model:
            model: LegacyMLPScorer | MemNetworkScorer = LegacyMLPScorer(
                input_dim=int(manifest.feature_dim),
                hidden_dims=manifest.hidden_dims,
                dropout=float(manifest.dropout),
            )
        else:
            suppress_memory_ids_path = base / "suppress_memory_ids.json"
            if suppress_memory_ids_path.exists():
                suppress_memory_ids = [str(item) for item in json.loads(suppress_memory_ids_path.read_text(encoding="utf-8"))]
                suppress_memory_id_to_index = {memory_id: idx + 1 for idx, memory_id in enumerate(suppress_memory_ids)}
            else:
                suppress_memory_ids = []
                suppress_memory_id_to_index = {}
            memory_id_vocab_size = max(
                2,
                int(manifest.memory_id_vocab_size),
                int(len(suppress_memory_ids) + 1),
            )
            model = MemNetworkScorer(
                embedding_dim=int(manifest.embedding_dim),
                state_dim=int(manifest.state_dim),
                num_hops=int(manifest.num_hops),
                num_heads=int(manifest.num_heads),
                memory_id_vocab_size=int(memory_id_vocab_size),
                type_vocab_size=int(manifest.type_vocab_size),
                lane_vocab_size=int(manifest.lane_vocab_size),
                type_emb_dim=int(manifest.type_emb_dim),
                lane_emb_dim=int(manifest.lane_emb_dim),
                dropout=float(manifest.dropout),
            )
        model_path = base / "model.pt"
        if model_path.exists():
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                try:
                    model.load_state_dict(state, strict=False)
                except RuntimeError:
                    current_state = model.state_dict()
                    filtered = {
                        key: value
                        for key, value in state.items()
                        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
                    }
                    model.load_state_dict(filtered, strict=False)
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

        memory_ids_path = base / "memory_ids.json"
        memory_vec_path = base / "memory_embeddings.npy"
        memory_texts_path = base / "memory_texts.jsonl"
        if memory_ids_path.exists() and memory_vec_path.exists():
            memory_ids = [str(item) for item in json.loads(memory_ids_path.read_text(encoding="utf-8"))]
            memory_matrix = np.asarray(np.load(memory_vec_path), dtype=np.float32)
        else:
            memory_ids = []
            memory_matrix = np.zeros((0, int(manifest.embedding_dim)), dtype=np.float32)
        memory_index_by_id = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}

        memory_text_by_id: dict[str, str] = {}
        if memory_texts_path.exists():
            for row in read_jsonl(memory_texts_path):
                memory_id = str(row.get("memory_id") or "")
                if memory_id:
                    memory_text_by_id[memory_id] = str(row.get("memory_text") or row.get("display_text") or "")

        feedback_store_path = base / "feedback_memory_store.jsonl"
        feedback_rows = list(read_jsonl(feedback_store_path)) if feedback_store_path.exists() else []
        suppress_memory_ids_path = base / "suppress_memory_ids.json"
        if suppress_memory_ids_path.exists():
            suppress_memory_ids = [str(item) for item in json.loads(suppress_memory_ids_path.read_text(encoding="utf-8"))]
            suppress_memory_id_to_index = {memory_id: idx + 1 for idx, memory_id in enumerate(suppress_memory_ids)}
        else:
            suppress_memory_ids, suppress_memory_id_to_index = build_memory_id_lookup(feedback_rows)
        feedback_query_matrix = (
            np.asarray(np.load(base / "feedback_query_embeddings.npy"), dtype=np.float32)
            if (base / "feedback_query_embeddings.npy").exists()
            else np.zeros((0, int(manifest.embedding_dim)), dtype=np.float32)
        )
        feedback_memory_matrix = (
            np.asarray(np.load(base / "feedback_memory_embeddings.npy"), dtype=np.float32)
            if (base / "feedback_memory_embeddings.npy").exists()
            else np.zeros((0, int(manifest.embedding_dim)), dtype=np.float32)
        )
        if feedback_query_matrix.shape != feedback_memory_matrix.shape:
            dim = int(manifest.embedding_dim)
            feedback_query_matrix = np.zeros((0, dim), dtype=np.float32)
            feedback_memory_matrix = np.zeros((0, dim), dtype=np.float32)
            feedback_rows = []
        if feedback_query_matrix.shape[0] != len(feedback_rows):
            n = min(feedback_query_matrix.shape[0], len(feedback_rows))
            feedback_query_matrix = feedback_query_matrix[:n]
            feedback_memory_matrix = feedback_memory_matrix[:n]
            feedback_rows = feedback_rows[:n]
        feedback_memory_id_ids_path = base / "feedback_memory_id_ids.npy"
        if feedback_memory_id_ids_path.exists():
            feedback_memory_id_ids = np.asarray(np.load(feedback_memory_id_ids_path), dtype=np.int64)
        else:
            feedback_memory_id_ids = np.asarray(
                [memory_id_to_index(str(row.get("m_id") or ""), suppress_memory_id_to_index) for row in feedback_rows],
                dtype=np.int64,
            )

        type_ids_path = base / "feedback_type_ids.npy"
        lane_ids_path = base / "feedback_lane_ids.npy"
        if type_ids_path.exists() and lane_ids_path.exists():
            feedback_type_ids = np.asarray(np.load(type_ids_path), dtype=np.int64)
            feedback_lane_ids = np.asarray(np.load(lane_ids_path), dtype=np.int64)
        else:
            feedback_type_ids, feedback_lane_ids = build_feedback_type_lane_arrays(feedback_rows)
        if (
            feedback_type_ids.shape[0] != len(feedback_rows)
            or feedback_lane_ids.shape[0] != len(feedback_rows)
            or feedback_memory_id_ids.shape[0] != len(feedback_rows)
        ):
            n = min(
                len(feedback_rows),
                int(feedback_type_ids.shape[0]),
                int(feedback_lane_ids.shape[0]),
                int(feedback_memory_id_ids.shape[0]),
            )
            feedback_rows = feedback_rows[:n]
            feedback_query_matrix = feedback_query_matrix[:n]
            feedback_memory_matrix = feedback_memory_matrix[:n]
            feedback_memory_id_ids = feedback_memory_id_ids[:n]
            feedback_type_ids = feedback_type_ids[:n]
            feedback_lane_ids = feedback_lane_ids[:n]
        if feedback_memory_id_ids.size == 0 and len(feedback_rows) > 0:
            feedback_memory_id_ids = np.asarray(
                [memory_id_to_index(str(row.get("m_id") or ""), suppress_memory_id_to_index) for row in feedback_rows],
                dtype=np.int64,
            )
        feedback_memory_id_ids = np.asarray(feedback_memory_id_ids, dtype=np.int64)
        feedback_memory_id_ids = np.clip(
            feedback_memory_id_ids,
            int(UNKNOWN_MEMORY_ID_INDEX),
            max(int(manifest.memory_id_vocab_size) - 1, int(UNKNOWN_MEMORY_ID_INDEX)),
        )

        return cls(
            artifact_dir=base,
            manifest=manifest,
            model=model,
            is_legacy_model=bool(is_legacy_model),
            encoder=encoder,
            memory_ids=memory_ids,
            memory_matrix=memory_matrix,
            memory_index_by_id=memory_index_by_id,
            memory_text_by_id=memory_text_by_id,
            feedback_rows=feedback_rows,
            feedback_query_matrix=feedback_query_matrix,
            feedback_memory_matrix=feedback_memory_matrix,
            feedback_memory_id_ids=feedback_memory_id_ids,
            feedback_type_ids=feedback_type_ids,
            feedback_lane_ids=feedback_lane_ids,
            suppress_memory_ids=suppress_memory_ids,
            suppress_memory_id_to_index=suppress_memory_id_to_index,
        )
