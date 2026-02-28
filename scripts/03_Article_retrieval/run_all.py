"""Master orchestration script for the Article Retrieval pipeline.

Runs all 11 planned experiments in a single pass by iterating the Cartesian
product of all ablation dimensions declared in the config.  Each combination
produces fully independent artifacts (indexes, embeddings, retrieval results,
metrics) thanks to dimension-encoded file names in config_utils.

Pipeline steps:
  00  build_article_index   — BM25 / TF-IDF / FAISS indexes
  01  build_query_dataset   — generate 24 query variations per link
  02  run_retrieval         — retrieve top-K articles per (retriever, version)
  03  run_reranking         — zero-shot re-rank with cross-encoder (Exp 6 & 7)
  04  train_reranker        — fine-tune cross-encoder on retrieval results (optional)
  05  evaluate              — compute Recall@K and MRR, write research CSV

Ablation dimensions swept automatically:
  Exp 1  — query versions (v1–v24)            [always, inner loop in step 02]
  Exp 2  — retriever models                   [always, inner loop in step 02]
  Exp 3  — corpus representation              [article_index.corpus_representations]
  Exp 4  — query context mode                 [queries.query_context_modes]
  Exp 5  — corpus granularity                 [article_index.corpus_granularities]
  Exp 6  — re-ranker model                    [reranking.models]
  Exp 7  — re-ranker input K                  [reranking.top_k_inputs]
  Exp 8  — domains                            [domains]
  Exp 9  — query sample size                  [queries.n_samples]
  Exp 11 — anchor preprocessing               [queries.anchor_preprocessings]
  (Exp 10 — FAISS index type — future work)

Usage:
  # Default run (all experiments, all domains):
  python scripts/03_Article_retrieval/run_all.py

  # Machine-specific config:
  python scripts/03_Article_retrieval/run_all.py --config configs/article_retrieval/kudremukh.yaml

  # Single domain:
  python scripts/03_Article_retrieval/run_all.py --domain money-heist

  # Subset of retrievers or versions:
  python scripts/03_Article_retrieval/run_all.py --retriever bm25 --versions 1,2,3

  # Skip already-done steps:
  python scripts/03_Article_retrieval/run_all.py --skip-index --skip-queries

  # Force rebuild of all cached artifacts:
  python scripts/03_Article_retrieval/run_all.py --force
"""
from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import article_retrieval.config_utils as cu

# ── Auto-select config by hostname ─────────────────────────────────────────────
_HOSTNAME_CONFIG: dict[str, str] = {
    "kudremukh": "configs/article_retrieval/kudremukh.yaml",
}
_DEFAULT_CONFIG = _HOSTNAME_CONFIG.get(
    socket.gethostname().lower(),
    "configs/article_retrieval/article_retrieval.yaml",
)


def _load_step(script_name: str):
    """Import a pipeline step script as a module, returning it."""
    path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(script_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_step(label: str, fn, *args, **kwargs) -> None:
    """Call a pipeline step function, print timing, exit on exception."""
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    t0 = time.time()
    fn(*args, **kwargs)
    print(f"  [OK] {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full article retrieval pipeline.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG,
                        help=f"Config file (auto-detected: {_DEFAULT_CONFIG}).")
    parser.add_argument("--domain", help="Single domain override.")
    parser.add_argument("--retriever", help="Comma-separated retriever names.")
    parser.add_argument("--versions",  help="Comma-separated version numbers.")
    parser.add_argument("--stage", choices=["retrieval", "reranking", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Force rebuild of all cached artifacts.")
    parser.add_argument("--skip-index",     action="store_true", help="Skip step 00.")
    parser.add_argument("--skip-queries",   action="store_true", help="Skip step 01.")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip step 02.")
    parser.add_argument("--skip-reranking", action="store_true", help="Skip step 03.")
    parser.add_argument("--skip-train",     action="store_true", help="Skip step 04 (reranker training).")
    parser.add_argument("--skip-eval",      action="store_true", help="Skip step 05.")
    args = parser.parse_args()

    # ── Load step modules ───────────────────────────────────────────────────────
    step00 = _load_step("00_build_article_index.py")
    step01 = _load_step("01_build_query_dataset.py")
    step02 = _load_step("02_run_retrieval.py")
    step03 = _load_step("03_run_reranking.py")
    step04 = _load_step("04_train_reranker.py")
    step05 = _load_step("05_evaluate.py")
    agg    = _load_step("aggregate_results.py")

    # ── Load config and expand ablation combinations ────────────────────────────
    base_config   = cu.load_config(ROOT / args.config)
    domains       = [args.domain] if args.domain else base_config.get("domains", [])
    ablation_cfgs = cu.get_ablation_configs(base_config)

    # ── Retriever / version overrides from CLI ──────────────────────────────────
    all_retrievers = (
        base_config.get("retrievers", {}).get("sparse", [])
        + base_config.get("retrievers", {}).get("dense", [])
    )
    retrievers = (
        [r.strip() for r in args.retriever.split(",")] if args.retriever else all_retrievers
    )
    cfg_versions = base_config.get("queries", {}).get("versions", list(range(1, 25)))
    versions = (
        [int(v.strip()) for v in args.versions.split(",")] if args.versions else cfg_versions
    )
    stages = ["retrieval", "reranking"] if args.stage == "all" else [args.stage]

    n_combos = len(ablation_cfgs)
    auto = " (auto-detected)" if args.config == _DEFAULT_CONFIG else " (user-supplied)"
    print(f"\nArticle Retrieval Pipeline — config: {args.config}{auto}")
    print(f"domains={domains}  ablation combos={n_combos}  "
          f"retrievers={len(retrievers)}  versions={len(versions)}  "
          f"stages={stages}  force={args.force}")

    t_total = time.time()

    for combo_idx, cfg in enumerate(ablation_cfgs, 1):
        label = cu.ablation_label(cfg)
        print(f"\n{'#' * 70}")
        print(f"  Combo {combo_idx}/{n_combos}: {label}")
        print(f"{'#' * 70}")

        for domain in domains:

            if not args.skip_index:
                _run_step(
                    f"00 — Build Article Index | {label}",
                    step00.build_for_domain, cfg, domain, args.force,
                )

            if not args.skip_queries:
                _run_step(
                    f"01 — Build Query Dataset | {label}",
                    step01.build_for_domain, cfg, domain, args.force,
                )

            if not args.skip_retrieval and "retrieval" in stages:
                _run_step(
                    f"02 — Run Retrieval | {label}",
                    step02.run_for_domain, cfg, domain, retrievers, versions, args.force,
                )

            if not args.skip_reranking and "reranking" in stages:
                if cfg.get("reranking", {}).get("enabled", True):
                    _run_step(
                        f"03 — Run Reranking | {label}",
                        step03.run_for_domain, cfg, domain, retrievers, versions, args.force,
                    )

            # Step 04: reranker training runs once (not per ablation combo) since
            # it uses a fixed source retriever and saves to a fixed checkpoint dir.
            # We only run it for the first combo to avoid redundant retraining.
            if not args.skip_train and combo_idx == 1:
                _run_step(
                    f"04 — Train Reranker | {domain}",
                    step04.train_for_domain, cfg, domain, args.force,
                )

            if not args.skip_eval:
                _run_step(
                    f"05 — Evaluate | {label}",
                    step05.evaluate_for_domain,
                    cfg, domain, retrievers, versions, stages, force=args.force,
                )

    # ── Final summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  All {n_combos} ablation combo(s) complete in {elapsed:.1f}s")
    print(f"  Research CSVs      : data/research/<domain>/")
    print(f"  Reranker checkpoint: data/article_retrieval/checkpoints/reranker_finetuned/")
    print(f"{'=' * 70}")

    # Print aggregate results across all domains
    for domain in domains:
        print(f"\n--- Aggregate results: {domain} ---")
        agg.print_table(
            [r for r in agg.load_research_csv(cu.get_research_csv_path(base_config, domain))
             if r.get("stage") == "retrieval"],
            "Retrieval Results (ranked by MRR)",
        )
        agg.print_table(
            [r for r in agg.load_research_csv(cu.get_research_csv_path(base_config, domain))
             if r.get("stage") == "reranking"],
            "Reranking Results (ranked by MRR)",
        )


if __name__ == "__main__":
    main()
