from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.association.graph_builder import AssociationBuildConfig, AssociationGraphBuilder
from agentmemory_v3.association.llm import AssociationLLM, AssociationLLMConfig
from agentmemory_v3.config import cfg_get, get_secret, load_yaml_config, resolve_path
from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig
from agentmemory_v3.utils.io import read_jsonl


def build_association_graph(
    config_path: str | Path,
    *,
    limit_memory: int = 0,
    max_l1_per_memory: int | None = None,
    hf_local_only: bool = False,
    hf_offline: bool = False,
    hf_online: bool = False,
) -> dict:
    cfg = load_yaml_config(config_path)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V5"))
    assoc_dir = resolve_path(cfg_get(cfg, "association.dir", "data/V5/association"))
    memory_rows = list(read_jsonl(root_dir / "processed" / "memory.jsonl"))
    if limit_memory > 0:
        memory_rows = memory_rows[:limit_memory]

    if hf_online:
        print("[v5][association] warning: --hf-online ignored; V5 E5 is forced local-only + offline")
    runtime_hf_local_only = True
    runtime_hf_offline = True

    encoder_cfg = SentenceEncoderConfig(
        model_name=str(cfg_get(cfg, "encoder.model_name", "intfloat/multilingual-e5-small")),
        use_e5_prefix=bool(cfg_get(cfg, "encoder.use_e5_prefix", True)),
        local_files_only=runtime_hf_local_only,
        offline=runtime_hf_offline,
        device=str(cfg_get(cfg, "encoder.device", "auto")),
        batch_size=int(cfg_get(cfg, "encoder.batch_size_cuda", cfg_get(cfg, "encoder.batch_size", 128))),
    )
    encoder = HFSentenceEncoder(
        encoder_cfg
    )
    runtime_llm_offline = bool(cfg_get(cfg, "association.llm.force_offline", False))
    if runtime_llm_offline:
        print("[v5][association] warning: LLM offline fallback mode enabled (quality will degrade)")
    api_key = ""
    if not runtime_llm_offline:
        api_key = get_secret(
            str(cfg_get(cfg, "association.llm.api_key_env", "DEEPSEEK_API_KEY")),
            env_file=str(cfg_get(cfg, "association.llm.env_file", "data/_secrets/deepseek.env")),
            default="",
        )
    llm_cfg = AssociationLLMConfig(
        api_key=api_key,
        base_url=str(cfg_get(cfg, "association.llm.base_url", "https://api.deepseek.com/v1")),
        model=str(cfg_get(cfg, "association.llm.model", "deepseek-chat")),
        timeout_s=float(cfg_get(cfg, "association.llm.timeout_s", 60.0)),
        temperature=float(cfg_get(cfg, "association.llm.temperature", 0.1)),
        allow_fallback=bool(cfg_get(cfg, "association.llm.allow_fallback", False)) or runtime_llm_offline,
        max_retries=int(cfg_get(cfg, "association.llm.max_retries", 3)),
        retry_backoff_s=float(cfg_get(cfg, "association.llm.retry_backoff_s", 1.0)),
        retry_max_backoff_s=float(cfg_get(cfg, "association.llm.retry_max_backoff_s", 8.0)),
    )
    llm = AssociationLLM(llm_cfg)
    builder = AssociationGraphBuilder(
        encoder=encoder,
        llm=llm,
        cfg=AssociationBuildConfig(
            output_dir=assoc_dir,
            merge_threshold=float(cfg_get(cfg, "association.merge_threshold", 0.96)),
            max_l1_per_memory=int(max_l1_per_memory or cfg_get(cfg, "association.max_l1_per_memory", 10)),
            l2_target_min=int(cfg_get(cfg, "association.l2_target_min", 64)),
            l2_target_max=int(cfg_get(cfg, "association.l2_target_max", 3000)),
            l3_target_min=int(cfg_get(cfg, "association.l3_target_min", 16)),
            l3_target_max=int(cfg_get(cfg, "association.l3_target_max", 500)),
            label_batch_size=int(cfg_get(cfg, "association.label_batch_size", 256)),
            bridge_top_k_per_node=int(cfg_get(cfg, "association.bridge_top_k_per_node", 20)),
            bridge_keep_ratio=float(cfg_get(cfg, "association.bridge_keep_ratio", 0.2)),
            bridge_min_weight=float(cfg_get(cfg, "association.bridge_min_weight", 0.18)),
            bridge_sem_weight=float(cfg_get(cfg, "association.bridge_sem_weight", 0.3)),
            bridge_co_weight=float(cfg_get(cfg, "association.bridge_co_weight", 0.7)),
            progress_every=int(cfg_get(cfg, "association.progress_every", 200)),
            llm_workers=int(cfg_get(cfg, "association.llm_workers", 16)),
            kmeans_use_minibatch=bool(cfg_get(cfg, "association.kmeans_use_minibatch", True)),
            kmeans_minibatch_threshold=int(cfg_get(cfg, "association.kmeans_minibatch_threshold", 12000)),
            kmeans_batch_size=int(cfg_get(cfg, "association.kmeans_batch_size", 2048)),
            kmeans_max_iter=int(cfg_get(cfg, "association.kmeans_max_iter", 120)),
            kmeans_random_state=int(cfg_get(cfg, "association.kmeans_random_state", 42)),
            cache_enabled=bool(cfg_get(cfg, "association.cache_enabled", True)),
            cache_dir_name=str(cfg_get(cfg, "association.cache_dir_name", "cache")),
            llm_l1_cache_file=str(cfg_get(cfg, "association.llm_l1_cache_file", "l1_extract_cache.jsonl")),
            llm_parent_seed_cache_file=str(cfg_get(cfg, "association.llm_parent_seed_cache_file", "parent_seed_cache.jsonl")),
            llm_cache_version=str(cfg_get(cfg, "association.llm_cache_version", "v1")),
            passage_embedding_manifest_file=str(
                cfg_get(cfg, "association.passage_embedding_manifest_file", "passage_embedding_manifest.json")
            ),
            passage_embedding_matrix_file=str(
                cfg_get(cfg, "association.passage_embedding_matrix_file", "passage_embedding_matrix.npy")
            ),
        ),
    )
    print(
        "[v5][association] encoder "
        f"model={encoder_cfg.model_name} local_only={encoder_cfg.local_files_only} offline={encoder_cfg.offline}"
    )
    print(
        "[v5][association] llm "
        f"mode={'offline_fallback' if runtime_llm_offline else 'online'} "
        f"model={llm_cfg.model}"
    )
    result = builder.build_and_save(memory_rows)
    return {"output_dir": str(assoc_dir), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5 association graph from memory.jsonl.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--limit-memory", type=int, default=0)
    parser.add_argument("--max-l1-per-memory", type=int, default=0)
    parser.add_argument("--hf-local-only", action="store_true")
    parser.add_argument("--hf-offline", action="store_true")
    parser.add_argument("--hf-online", action="store_true")
    args = parser.parse_args()

    result = build_association_graph(
        args.config,
        limit_memory=args.limit_memory,
        max_l1_per_memory=args.max_l1_per_memory if args.max_l1_per_memory > 0 else None,
        hf_local_only=bool(args.hf_local_only),
        hf_offline=bool(args.hf_offline),
        hf_online=bool(args.hf_online),
    )
    print(
        f"[v5][association] graph built -> {result['output_dir']} "
        f"L1={result['l1_count']} L2={result['l2_count']} L3={result['l3_count']} "
        f"parent_edges={result['parent_edge_count']} bridge_edges={result['bridge_edge_count']}"
    )


if __name__ == "__main__":
    main()
