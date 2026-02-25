"""Cross-domain corpus summary.

Reads every data/stats/<domain>.json and prints a consolidated view of:
  - Scraping coverage
  - Dataset size (articles, paragraphs, sentences, links)
  - Split sizes (if span ID splits have been built)
  - Best model results (if evaluation has been run)

Also optionally saves a machine-readable summary to data/stats/corpus_summary.json.

Run:
  python scripts/summarise_corpus.py
  python scripts/summarise_corpus.py --save
  python scripts/summarise_corpus.py --domain beverlyhillscop
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
STATS_DIR = ROOT / "data" / "stats"


def load_all(domain_filter: str | None) -> list[dict]:
    files = sorted(STATS_DIR.glob("*.json"))
    # Skip the summary file itself
    files = [f for f in files if f.stem != "corpus_summary"]
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if domain_filter and data.get("domain") != domain_filter:
            continue
        rows.append(data)
    return rows


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{100 * n / total:5.1f}%"


def print_scraping(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("  SCRAPING COVERAGE")
    print("=" * 78)
    print(f"  {'Domain':<22} {'Total':>6} {'Downloaded':>11} {'Skipped':>8} "
          f"{'Failed':>7} {'Coverage':>9} {'HTML MB':>8} {'Text KB':>8}")
    print("  " + "-" * 74)
    for d in rows:
        sc = d.get("scraping", {})
        if not sc:
            continue
        dom  = d.get("domain", "?")
        tot  = sc.get("total_urls", 0)
        dl   = sc.get("downloaded", 0)
        sk   = sc.get("skipped", 0)
        fail = sc.get("failed", 0)
        html_mb = sc.get("html_bytes", {}).get("total", 0) / 1_048_576
        text_kb = sc.get("text_bytes", {}).get("total", 0) / 1_024
        print(f"  {dom:<22} {tot:>6} {dl:>11} {sk:>8} {fail:>7} "
              f"{_pct(dl, tot):>9} {html_mb:>8.1f} {text_kb:>8.0f}")


def print_dataset(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("  DATASET STATISTICS")
    print("=" * 78)
    print(f"  {'Domain':<22} {'Articles':>9} {'Paragraphs':>11} {'Sentences':>10} "
          f"{'Int.Links':>10} {'Avg.Len':>8} {'Avg.Lnk':>8}")
    print("  " + "-" * 74)

    total_art = total_para = total_sent = total_int = 0
    for d in rows:
        ds = d.get("dataset_stats", {})
        if not ds:
            continue
        dom  = d.get("domain", "?")
        art  = ds.get("num_articles", 0)
        para = ds.get("num_paragraphs", 0)
        sent = ds.get("num_sentences", 0)
        intl = ds.get("link_type_counts", {}).get("internal", 0)
        avg_len = ds.get("avg_article_length_chars", 0)
        avg_lnk = ds.get("avg_internal_links_per_article", 0.0)
        print(f"  {dom:<22} {art:>9,} {para:>11,} {sent:>10,} "
              f"{intl:>10,} {avg_len:>8,} {avg_lnk:>8.2f}")
        total_art  += art
        total_para += para
        total_sent += sent
        total_int  += intl

    if len(rows) > 1:
        print("  " + "-" * 74)
        print(f"  {'TOTAL':<22} {total_art:>9,} {total_para:>11,} {total_sent:>10,} "
              f"{total_int:>10,}")


def print_splits(rows: list[dict]) -> None:
    has_splits = any(d.get("dataset_stats", {}).get("split_sizes") for d in rows)
    if not has_splits:
        return

    print("\n" + "=" * 78)
    print("  TRAIN / VAL / TEST SPLITS")
    print("=" * 78)
    print(f"  {'Domain':<22} {'Granularity':<12} {'Train':>7} {'Val':>6} {'Test':>6}")
    print("  " + "-" * 60)
    for d in rows:
        ds = d.get("dataset_stats", {})
        splits = ds.get("split_sizes", {})
        if not splits:
            continue
        dom = d.get("domain", "?")
        for gran, sz in sorted(splits.items()):
            print(f"  {dom:<22} {gran:<12} "
                  f"{sz.get('train', 0):>7,} {sz.get('val', 0):>6,} {sz.get('test', 0):>6,}")


def print_best_results(rows: list[dict]) -> None:
    has_results = any(
        d.get("span_id") or d.get("article_retrieval") or d.get("linking_pipeline")
        for d in rows
    )
    if not has_results:
        return

    print("\n" + "=" * 78)
    print("  BEST MODEL RESULTS")
    print("=" * 78)

    for d in rows:
        dom = d.get("domain", "?")
        sections = []

        sid = d.get("span_id", {}).get("overall_best")
        if sid:
            sections.append(
                f"  Span ID     model={sid.get('model','?').split('/')[-1]:<20} "
                f"gran={sid.get('granularity','?'):<10} "
                f"scheme={sid.get('label_scheme','?'):<6} "
                f"span_f1={sid.get('span_f1', 0):.4f}"
            )

        ar = d.get("article_retrieval", {})
        br = ar.get("best_retrieval")
        brr = ar.get("best_reranking")
        if br:
            sections.append(
                f"  Retrieval   retriever={br.get('retriever','?').split('/')[-1]:<18} "
                f"v{br.get('version','?'):<3} "
                f"R@1={br.get('recall_at_1', 0):.3f}  "
                f"MRR={br.get('mrr', 0):.4f}"
            )
        if brr:
            sections.append(
                f"  Reranking   retriever={brr.get('retriever','?').split('/')[-1]:<18} "
                f"v{brr.get('version','?'):<3} "
                f"R@1={brr.get('recall_at_1', 0):.3f}  "
                f"MRR={brr.get('mrr', 0):.4f}"
            )

        lp = d.get("linking_pipeline", {}).get("best")
        if lp:
            sections.append(
                f"  Linking     retriever={lp.get('retriever','?').split('/')[-1]:<18} "
                f"stage={lp.get('stage','?'):<10} "
                f"link_f1={lp.get('linking_f1', 0):.4f}  "
                f"ent_acc={lp.get('entity_accuracy', 0):.4f}"
            )

        if sections:
            print(f"\n  [{dom}]")
            for s in sections:
                print(s)


def build_summary(rows: list[dict]) -> dict:
    """Build a machine-readable corpus_summary dict."""
    domains = []
    totals = {
        "num_domains": 0,
        "num_articles": 0,
        "num_paragraphs": 0,
        "num_sentences": 0,
        "num_internal_links": 0,
    }
    for d in rows:
        ds = d.get("dataset_stats", {})
        dom_entry: dict = {
            "domain":         d.get("domain", "?"),
            "scrape_coverage": None,
        }
        sc = d.get("scraping", {})
        if sc.get("total_urls", 0) > 0:
            dom_entry["scrape_coverage"] = round(
                sc["downloaded"] / sc["total_urls"], 4
            )
        dom_entry["dataset_stats"]     = ds
        dom_entry["span_id"]           = d.get("span_id")
        dom_entry["article_retrieval"] = d.get("article_retrieval")
        dom_entry["linking_pipeline"]  = d.get("linking_pipeline")
        domains.append(dom_entry)

        totals["num_domains"]        += 1
        totals["num_articles"]       += ds.get("num_articles", 0)
        totals["num_paragraphs"]     += ds.get("num_paragraphs", 0)
        totals["num_sentences"]      += ds.get("num_sentences", 0)
        totals["num_internal_links"] += ds.get("link_type_counts", {}).get("internal", 0)

    return {
        "generated_at": datetime.now().isoformat(),
        "totals":        totals,
        "domains":       domains,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain corpus summary.")
    parser.add_argument("--domain",  help="Show a single domain only.")
    parser.add_argument("--save",    action="store_true",
                        help="Save summary to data/stats/corpus_summary.json.")
    parser.add_argument("--json",    action="store_true",
                        help="Print machine-readable JSON instead of tables.")
    args = parser.parse_args()

    rows = load_all(args.domain)
    if not rows:
        print(f"No stats files found in {STATS_DIR}")
        return

    if args.json:
        print(json.dumps(build_summary(rows), indent=2))
    else:
        print_scraping(rows)
        print_dataset(rows)
        print_splits(rows)
        print_best_results(rows)
        print()

    if args.save:
        summary = build_summary(rows)
        out = STATS_DIR / "corpus_summary.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Summary saved → {out}")


if __name__ == "__main__":
    main()
