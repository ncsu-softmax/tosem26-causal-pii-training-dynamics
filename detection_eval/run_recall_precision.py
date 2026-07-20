#!/usr/bin/env python
"""
Recall & precision of the DETECTION stage (regex + StarPII NER, NO LLM refine)
on the human-annotated ground-truth set (808 unique files, 1,689 PII spans).

This reproduces the per-type recall/precision reported in the paper
(Section 5.2, Table 3: "Detection Performance by Type: Regex vs StarPII vs Combined").

It runs detection once and writes, in a single consistent pass:
    results/metrics_table.md      - the Table 3 numbers (regex / StarPII / combined R & P)
    results/metrics.json          - the same numbers as machine-readable JSON
    results/samples.md            - sampled TP / FN / FP spans with context
    results/false_positives.jsonl - every combined FP (for manual inspection)
    results/missed_combined.jsonl - every GT span the combined detector missed

The detectors are imported from the project source (./src), so this is byte-for-byte
the same detection code the dataset-construction pipeline uses:
    - regex   : src/pii_regex.py  find_pii_by_regex          (email, ip_address, key)
    - StarPII : src/utils.py      get_pipeline_ner / extract_pii_from_chunks_batch  (all 6 types)

Environment
    Needs the project env (see ../requirement.txt: torch + transformers) and a HuggingFace
    token with access to `bigcode/starpii`. Provide the token either via the HF_TOKEN
    environment variable or a `.env` file at the repository root (HF_TOKEN=hf_xxx).

Run (from the repository root or this folder):
    python detection_eval/run_recall_precision.py

Definitions
    - Match rule = same pii_type AND any character overlap.
    - Recall    = (GT spans with >=1 overlapping same-type prediction) / (GT spans).   [GT-centric]
    - Precision = (predictions overlapping >=1 same-type GT span) / (predictions).      [pred-centric]

Precision caveat: the GT is a CURATED, non-exhaustive set, so precision is NOT a clean
detector precision (a "false positive" may be a genuine PII span that was simply not part of
the curated GT). Recall is the trustworthy metric.
"""

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ----- paths & project imports ---------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Load HF_TOKEN from a repo-root .env if python-dotenv is available; otherwise fall
# back to whatever is already in the environment.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
except ImportError:
    pass
HF_TOKEN = os.environ.get("HF_TOKEN")

DATA = HERE / "data" / "groundtruth_pii_flat_min150.jsonl"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PII_TYPES = ["email", "password", "name", "username", "ip_address", "key"]
DETECTORS = ["regex", "starpii", "combined"]
# PII types covered by the regex detector (mirrors `regex_match` in src/search.py).
REGEX_TYPES = ["email", "ip_address", "key"]
SEED = 0
N_SAMPLE = 8  # samples per type per category in samples.md

# Expected ground-truth shape (matches paper Section 5.2 / Table 3).
EXPECTED_FILES = 808
EXPECTED_SPANS = 1689


# ----- load & group --------------------------------------------------------------
def load_files():
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    files = {}  # fid -> {content, gt: {(type,start,end): {value, source}}}
    for r in rows:
        fid = str(r["file_id"])
        f = files.setdefault(fid, {"content": r.get("content") or "", "gt": {}})
        key = (r["pii_type"], r["location_start"], r["location_end"])
        f["gt"].setdefault(key, {"value": r["value"], "source": r.get("source", "")})
    return files


def overlaps(s, e, s2, e2):
    return s < e2 and s2 < e


def ctx(content, s, e, w=55):
    return " ".join(
        (
            content[max(0, s - w) : s] + "[[" + content[s:e] + "]]" + content[e : e + w]
        ).split()
    )


# ----- detection -----------------------------------------------------------------
def run_regex(files, file_ids):
    from pii_regex import find_pii_by_regex

    spans = {}
    for fid in file_ids:
        out = []
        for pt in REGEX_TYPES:
            try:
                for r in find_pii_by_regex(files[fid]["content"], pt):
                    out.append((r["piiType"], r["location_start"], r["location_end"]))
            except Exception as ex:
                print("regex error", pt, fid, ex)
        spans[fid] = list(dict.fromkeys(out))
    return spans


def run_starpii(files, file_ids):
    from utils import (
        get_pipeline_ner,
        map_starpii_to_pii_types,
        extract_pii_from_chunks_batch,
    )

    if not HF_TOKEN:
        print(
            "WARNING: HF_TOKEN not set. `bigcode/starpii` is a gated model; if the "
            "environment is not already logged in to HuggingFace this will fail. "
            "Set HF_TOKEN in the environment or a repo-root .env file."
        )
    nlp, tok, max_len = get_pipeline_ner()
    contents = [files[fid]["content"] for fid in file_ids]
    ner = extract_pii_from_chunks_batch(contents, nlp, tok, max_len, batch_size=64)
    spans = {}
    for fid, results in zip(file_ids, ner):
        out = []
        for r in results:
            m = map_starpii_to_pii_types.get(r["entity_group"])
            if m:
                out.append((m, r["start"], r["end"]))
        spans[fid] = list(dict.fromkeys(out))
    return spans


# ----- scoring -------------------------------------------------------------------
def score_recall(files, file_ids, pred):
    """GT-centric. Returns tp, fn (Counters) and the list of missed GT records."""
    tp, fn, missed = Counter(), Counter(), []
    for fid in file_ids:
        by_type = defaultdict(list)
        for t, s, e in pred.get(fid, []):
            by_type[t].append((s, e))
        for (t, s, e), meta in files[fid]["gt"].items():
            if any(overlaps(s, e, ps, pe) for (ps, pe) in by_type[t]):
                tp[t] += 1
            else:
                fn[t] += 1
                missed.append(
                    {
                        "file_id": fid,
                        "pii_type": t,
                        "value": meta["value"],
                        "source": meta["source"],
                        "location_start": s,
                        "location_end": e,
                        "context": ctx(files[fid]["content"], s, e),
                    }
                )
    return tp, fn, missed


def score_precision(files, file_ids, pred):
    """Pred-centric. Returns tp, fp (Counters) and the list of FP records."""
    tp, fp, fps = Counter(), Counter(), []
    for fid in file_ids:
        gt_by_type = defaultdict(list)
        for t, s, e in files[fid]["gt"]:
            gt_by_type[t].append((s, e))
        for t, s, e in pred.get(fid, []):
            if any(overlaps(s, e, gs, ge) for (gs, ge) in gt_by_type[t]):
                tp[t] += 1
            else:
                fp[t] += 1
                fps.append(
                    {
                        "file_id": fid,
                        "pii_type": t,
                        "value": files[fid]["content"][s:e],
                        "location_start": s,
                        "location_end": e,
                        "context": ctx(files[fid]["content"], s, e),
                    }
                )
    return tp, fp, fps


def recall_tp_records(files, file_ids, pred):
    """GT records that WERE found (for TP sampling in samples.md)."""
    found = []
    for fid in file_ids:
        by_type = defaultdict(list)
        for t, s, e in pred.get(fid, []):
            by_type[t].append((s, e))
        for (t, s, e), meta in files[fid]["gt"].items():
            if any(overlaps(s, e, ps, pe) for (ps, pe) in by_type[t]):
                found.append(
                    {
                        "file_id": fid,
                        "pii_type": t,
                        "value": meta["value"],
                        "location_start": s,
                        "location_end": e,
                        "context": ctx(files[fid]["content"], s, e),
                    }
                )
    return found


# ----- output writers ------------------------------------------------------------
def fmt(x):
    return f"{x:.3f}" if x is not None else "-"


def recall_of(rec, t):
    g = rec["tp"][t] + rec["fn"][t]
    return rec["tp"][t] / g if g else None


def prec_of(prc, t):
    d = prc["tp"][t] + prc["fp"][t]
    return prc["tp"][t] / d if d else None


def overall_recall(rec):
    tt = sum(rec["tp"].values())
    gg = tt + sum(rec["fn"].values())
    return tt / gg if gg else None


def overall_prec(prc):
    tt = sum(prc["tp"].values())
    dd = tt + sum(prc["fp"].values())
    return tt / dd if dd else None


def write_metrics_table(files, rec, prc, recall_by_source):
    gt_count = Counter()
    for f in files.values():
        for t, _, _ in f["gt"]:
            gt_count[t] += 1
    L = [
        "# Metrics - recall & precision by PII type",
        "",
        f"Dataset: `data/{DATA.name}`  |  unique files: **{len(files)}**  |  "
        f"GT spans: **{sum(gt_count.values())}**",
        "",
        "Detection = regex (email/ip_address/key) + StarPII NER (all types); **no LLM refine**.  ",
        "Match = same type + any character overlap.  ",
        "Recall is GT-centric; precision is prediction-centric.",
        "",
        "## Recall (R) and Precision (P)",
        "",
        "| type | GT | regex R | regex P | StarPII R | StarPII P | combined R | combined P |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in PII_TYPES:
        L.append(
            f"| {t} | {gt_count[t]} | "
            f"{fmt(recall_of(rec['regex'], t))} | {fmt(prec_of(prc['regex'], t))} | "
            f"{fmt(recall_of(rec['starpii'], t))} | {fmt(prec_of(prc['starpii'], t))} | "
            f"{fmt(recall_of(rec['combined'], t))} | {fmt(prec_of(prc['combined'], t))} |"
        )
    L.append(
        f"| **overall** | **{sum(gt_count.values())}** | "
        f"{fmt(overall_recall(rec['regex']))} | {fmt(overall_prec(prc['regex']))} | "
        f"{fmt(overall_recall(rec['starpii']))} | {fmt(overall_prec(prc['starpii']))} | "
        f"{fmt(overall_recall(rec['combined']))} | {fmt(overall_prec(prc['combined']))} |"
    )
    L += [
        "",
        "`-` = regex emits no predictions for that type (covers only email/ip_address/key).",
    ]
    # Optional robustness check: only shown if the ground truth carries `source` labels
    # distinguishing the original set from a later recall audit.
    if recall_by_source["recall_audit"]["tp"] + recall_by_source["recall_audit"]["fn"]:
        L += [
            "",
            "## Combined recall by GT source (robustness check)",
            "",
            "| group | GT | found | recall |",
            "|---|---|---|---|",
        ]
        for grp in ("original", "recall_audit"):
            d = recall_by_source[grp]
            g = d["tp"] + d["fn"]
            L.append(f"| {grp} | {g} | {d['tp']} | {fmt(d['tp'] / g if g else None)} |")
    (RESULTS / "metrics_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_samples(files, rec_missed, fps, tp_found):
    rng = random.Random(SEED)
    L = [
        "# Sampled real results (combined = regex u StarPII)",
        "",
        f"Random samples (seed={SEED}), up to {N_SAMPLE} per type. `[[...]]` marks the span.",
        "",
        "Legend: **TP** = GT correctly found - **FN** = GT missed - "
        "**FP** = prediction not matching any same-type GT.",
        "",
    ]

    def block(title, items, show_source=False):
        L.append(f"## {title}")
        by_type = defaultdict(list)
        for m in items:
            by_type[m["pii_type"]].append(m)
        for t in PII_TYPES:
            pool = by_type.get(t, [])
            if not pool:
                continue
            L.append(f"\n### {t} ({len(pool)})")
            for m in rng.sample(pool, min(N_SAMPLE, len(pool))):
                extra = f"  _src={m.get('source','')}_" if show_source else ""
                L.append(f"- `{m['value']}`{extra}")
                L.append(f"  - {m['context']}")
        L.append("")

    block("TP - ground truth correctly detected", tp_found)
    block("FN - ground truth missed by detection", rec_missed, show_source=True)
    block("FP - predictions not matching ground truth", fps)
    (RESULTS / "samples.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    print("loading dataset:", DATA)
    files = load_files()
    file_ids = list(files)
    gt_count = Counter()
    for f in files.values():
        for t, _, _ in f["gt"]:
            gt_count[t] += 1
    print(
        f"files={len(file_ids)}  GT spans={sum(gt_count.values())}  by_type={dict(gt_count)}"
    )

    # ---- integrity checks (fail fast if the dataset is not what we expect) ----
    if len(file_ids) != EXPECTED_FILES:
        print(
            f"WARNING: expected {EXPECTED_FILES} files, got {len(file_ids)} "
            "(metrics are per-span and unaffected, but the file count in the paper "
            "should be reconciled)."
        )
    assert (
        sum(gt_count.values()) == EXPECTED_SPANS
    ), f"expected {EXPECTED_SPANS} GT spans, got {sum(gt_count.values())}"
    bad = sum(
        1
        for f in files.values()
        for (t, s, e), meta in f["gt"].items()
        if f["content"][s:e] != meta["value"]
    )
    if bad:
        raise SystemExit(f"ABORT: {bad} GT spans have content[start:end] != value")
    print(f"integrity OK: {len(file_ids)} files, {sum(gt_count.values())} spans, all offsets valid")

    t0 = time.time()
    regex_spans = run_regex(files, file_ids)
    print(
        f"regex done in {time.time()-t0:.1f}s  spans={sum(len(v) for v in regex_spans.values())}"
    )
    t0 = time.time()
    starpii_spans = run_starpii(files, file_ids)
    print(
        f"starpii done in {time.time()-t0:.1f}s  spans={sum(len(v) for v in starpii_spans.values())}"
    )
    combined_spans = {
        fid: list(set(regex_spans[fid]) | set(starpii_spans[fid])) for fid in file_ids
    }

    pred = {"regex": regex_spans, "starpii": starpii_spans, "combined": combined_spans}
    rec, prc = {}, {}
    missed = {}
    fps = {}
    for name in DETECTORS:
        tp, fn, ms = score_recall(files, file_ids, pred[name])
        rec[name] = {"tp": tp, "fn": fn}
        missed[name] = ms
        ptp, pfp, pf = score_precision(files, file_ids, pred[name])
        prc[name] = {"tp": ptp, "fp": pfp}
        fps[name] = pf

    # recall by GT source group (combined) - only meaningful if the GT has source labels
    recall_by_source = {
        "original": {"tp": 0, "fn": 0},
        "recall_audit": {"tp": 0, "fn": 0},
    }
    for fid in file_ids:
        by_type = defaultdict(list)
        for t, s, e in combined_spans[fid]:
            by_type[t].append((s, e))
        for (t, s, e), meta in files[fid]["gt"].items():
            grp = "recall_audit" if meta["source"] == "recall_audit" else "original"
            hit = any(overlaps(s, e, ps, pe) for (ps, pe) in by_type[t])
            recall_by_source[grp]["tp" if hit else "fn"] += 1

    # ---- write outputs ----
    write_metrics_table(files, rec, prc, recall_by_source)
    tp_found = recall_tp_records(files, file_ids, combined_spans)
    write_samples(files, missed["combined"], fps["combined"], tp_found)

    with open(RESULTS / "false_positives.jsonl", "w", encoding="utf-8") as f:
        for m in fps["combined"]:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(RESULTS / "missed_combined.jsonl", "w", encoding="utf-8") as f:
        for m in missed["combined"]:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    metrics = {
        "n_files": len(file_ids),
        "n_gt_spans": sum(gt_count.values()),
        "gt_by_type": dict(gt_count),
        "pred_counts": {n: sum(len(v) for v in pred[n].values()) for n in DETECTORS},
        "recall": {
            n: {
                "by_type": {t: recall_of(rec[n], t) for t in PII_TYPES},
                "overall": overall_recall(rec[n]),
            }
            for n in DETECTORS
        },
        "precision": {
            n: {
                "by_type": {t: prec_of(prc[n], t) for t in PII_TYPES},
                "overall": overall_prec(prc[n]),
            }
            for n in DETECTORS
        },
        "recall_by_source_combined": recall_by_source,
        "combined_fp_total": len(fps["combined"]),
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== combined recall/precision overall ===")
    print("recall   :", fmt(overall_recall(rec["combined"])))
    print("precision:", fmt(overall_prec(prc["combined"])))
    print(
        "\nwrote: metrics_table.md, samples.md, metrics.json, "
        "false_positives.jsonl, missed_combined.jsonl  (in results/)"
    )


if __name__ == "__main__":
    main()
