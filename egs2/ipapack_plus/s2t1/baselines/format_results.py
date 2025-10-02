"""Format results from multiple metrics.json files into a single CSV.
Usage:
    python baselines/format_results.py \
        --path_pattern "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/preds/*/metrics.json" \
        --output /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/preds/outputs.csv
"""

import json, csv, glob, argparse

COLUMNS = ["model", "dataset", "PFER", "FER", "FED", "PER", "N"]


def fmt(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path_pattern",
        required=True,
        help="Path pattern to search for metrics.json files",
    )
    ap.add_argument("--output", required=True, help="Path to save output CSV")
    args = ap.parse_args()

    allpaths = glob.glob(args.path_pattern)
    rows = []
    for f in allpaths:
        with open(f) as fh:
            data = json.load(fh)
            rows.append({k: fmt(data.get(k, None)) for k in COLUMNS})

    if not rows:
        return

    with open(args.output, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
