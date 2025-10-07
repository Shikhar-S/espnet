"""Run evaluation with baselines.
Usage:

For evaluation we need panphon 0.22. Run with:
    python baselines/evaluate.py \
        --dataset buckeye \
        --model 'facebook/wav2vec2-lv-60-espeak-cv-ft'

Available models:
'facebook/wav2vec2-lv-60-espeak-cv-ft' 'facebook/wav2vec2-xlsr-53-espeak-cv-ft' \
'ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns' 'allophant' 'allosaurus'

Available datasets:
'aishell' 'buckeye' 'cv' 'doreco' 'epadb' 'fleurs' \
'fleurs_indv' 'kazakh' 'l2arctic' 'librispeech' \
'mls_dutch' 'mls_french' 'mls_german' \
'mls_italian' 'mls_polish' 'mls_portuguese' \
'mls_spanish' 'southengland' 'speechoceannotth' \
'tamil' 'tusom2021' 'voxangeles'
"""

import argparse
import os
import string
import unicodedata
import panphon.distance
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {file}")


def clean_text(s):
    """Normalize IPA text: remove spaces/punct, NFC->NFD, fix 'g'→'ɡ'"""
    s = s.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
    s = unicodedata.normalize("NFD", s)
    return s.replace("g", "ɡ").strip()


def compute_chunk_metrics(chunk):
    """Compute metrics for a chunk of (key, hyp, ref) tuples"""
    dst = panphon.distance.Distance()
    metrics = {}
    pfer_sum = fed_sum = per_err = phones = 0

    for key, h, r in tqdm(chunk, desc="Processing", leave=False):
        pfer = dst.hamming_feature_edit_distance(h, r)
        fed = dst.feature_edit_distance(h, r)
        per_errors = dst.min_edit_distance(
            lambda v: 1,
            lambda v: 1,
            lambda x, y: 0 if x == y else 1,
            [[]],
            dst.fm.ipa_segs(h),
            dst.fm.ipa_segs(r),
        )
        n_phones = len(dst.fm.ipa_segs(r))

        metrics[key] = {
            "pfer": pfer,
            "fed": fed,
            "per": (per_errors / n_phones * 100) if n_phones > 0 else 0.0,
            "fer": (fed / n_phones * 100) if n_phones > 0 else 0.0,
        }

        pfer_sum += pfer
        fed_sum += fed
        per_err += per_errors
        phones += n_phones

    return metrics, pfer_sum, fed_sum, per_err, phones, len(chunk)


def compute_metrics(test_data, num_workers=1):
    """Compute aggregate and instance metrics"""
    # Prepare items
    items = [
        (k, clean_text(v["prediction"]), clean_text(v["transcription"]))
        for k, v in test_data.items()
    ]

    if not items:
        return {"PFER": 0, "FER": 0, "FED": 0, "PER": 0, "N": 0}, {}

    # Parallel processing
    chunk_size = max(1, len(items) // num_workers)
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(compute_chunk_metrics, chunks),
                total=len(chunks),
                desc="Computing metrics",
            )
        )

    # Combine results
    instance_metrics = {}
    for r in results:
        instance_metrics.update(r[0])

    pfer_sum = sum(r[1] for r in results)
    fed_sum = sum(r[2] for r in results)
    per_err = sum(r[3] for r in results)
    phones = sum(r[4] for r in results)
    n = len(items)

    aggregate = {
        "PFER": pfer_sum / n,
        "FER": (fed_sum / phones * 100) if phones > 0 else 0,
        "FED": fed_sum,
        "PER": (per_err / phones * 100) if phones > 0 else 0,
        "N": n,
    }

    return aggregate, instance_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default="./preds")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    base = f"{args.output_dir}/{args.dataset}.{args.model.replace('/','.')}"
    os.makedirs(base, exist_ok=True)

    # Load predictions
    pred_file = f"{base}/preds.json"
    print(f"Loading: {pred_file}")
    test_data = load_json(pred_file)
    print(f"Loaded {len(test_data)} utterances")

    # Compute metrics
    agg, inst = compute_metrics(test_data, args.workers)

    # Display results
    print(f"\n{args.model} on {args.dataset}")
    print("=" * 40)
    for k, v in agg.items():
        print(f"{k}: {v:.2f}")
    print("=" * 40)

    # Save
    save_json(
        {**agg, "model": args.model, "dataset": args.dataset}, f"{base}/metrics.json"
    )
    save_json(inst, f"{base}/instance_metrics.json")


if __name__ == "__main__":
    main()
