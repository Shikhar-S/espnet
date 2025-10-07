"""Upload prediction to zeno for visualization and analysis.
Usage:
    python baselines/zeno_upload.py \
        --pattern "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/preds/buckeye.*/preds.json" \
        --api-key zen_PKS1WveBwIsxT1AK0vNauaQFYYVAL92zTeOBxCSrIPc \
        --project-name powsm \
        --num-samples 200 \
        --seed 42 \
        --sampled_items "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/sampled_items/buckeye.txt"
"""

import json
import argparse
from pathlib import Path
from glob import glob
import pandas as pd
from zeno_client import ZenoClient, ZenoMetric
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern", required=True, help="Glob pattern for preds.json files"
    )
    parser.add_argument("--api-key", required=True, help="Zeno API key")
    parser.add_argument("--project-name", required=True, help="Zeno project name")
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Samples per dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sampled_items", type=str, default=None, help="Path to save sampled item keys"
    )
    return parser.parse_args()


def extract_model_dataset(filepath):
    """Extract model and dataset from path: .../dataset.model/preds.json"""
    parent = Path(filepath).parent.name
    parts = parent.split(".")
    return parts[0], ".".join(parts[1:]) if len(parts) > 1 else "unknown"


def load_data(pattern, num_samples=None, seed=42, sampled_items_path=None):
    """Load predictions and metrics"""
    data = {}

    for pred_file in glob(pattern):
        dataset, model = extract_model_dataset(pred_file)
        metrics_file = str(Path(pred_file).parent / "instance_metrics.json")

        if not Path(metrics_file).exists():
            print(f"Skipping {model} on {dataset}: no instance_metrics.json found")
            continue

        with open(pred_file) as f:
            preds = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Combine predictions and metrics
        combined = {}
        for key in preds:
            if key in metrics:
                combined[key] = {**preds[key], **metrics[key]}

        if dataset not in data:
            data[dataset] = {}
        data[dataset][model] = combined

    # Sample if requested
    if num_samples or sampled_items_path is not None:
        random.seed(seed)
        for dataset in data:
            first_model = next(iter(data[dataset].values()))
            all_keys = list(first_model.keys())
            if sampled_items_path is not None:
                with open(sampled_items_path, "r") as f:
                    sampled_keys = set(line.strip() for line in f)
                print(
                    f"{dataset}: using {len(sampled_keys)}/{len(all_keys)} from {sampled_items_path}"
                )
            else:
                sampled_keys = set(
                    random.sample(all_keys, min(num_samples, len(all_keys)))
                )
                print(
                    f"{dataset}: randomly sampled {len(sampled_keys)}/{len(all_keys)}"
                )

            for model in data[dataset]:
                data[dataset][model] = {
                    k: v for k, v in data[dataset][model].items() if k in sampled_keys
                }
            print(f"{dataset}: sampled {len(sampled_keys)}/{len(all_keys)}")

    return data


def upload_to_zeno(data, api_key, project_name):
    """Upload datasets and models to Zeno"""
    client = ZenoClient(api_key)
    S3_BASE = "https://l2arctic.s3.us-east-2.amazonaws.com"

    for dataset, models in data.items():
        print(f"\nDataset: {dataset}")

        # Base dataset
        first_model = next(iter(models.values()))
        examples = list(first_model.items())

        base_df = pd.DataFrame(
            {
                "id": [k for k, _ in examples],
                # "data": [
                #     S3_BASE + "/" + Path(ex["wavpath"]).name for _, ex in examples
                # ],
                "label": [ex["transcription"] for _, ex in examples],
                "data": [S3_BASE + "/" + k + ".flac" for k, _ in examples],
            }
        )

        # Create project
        project = client.create_project(
            name=f"{project_name}-{dataset}",
            view="audio-transcription",
            metrics=[
                ZenoMetric(name="pfer", type="mean", columns=["pfer"]),
                ZenoMetric(name="fer", type="mean", columns=["fer"]),
                ZenoMetric(name="per", type="mean", columns=["per"]),
            ],
        )

        project.upload_dataset(
            base_df, id_column="id", data_column="data", label_column="label"
        )

        # Upload models
        for model_name, predictions in models.items():
            print(f"  Model: {model_name}")
            examples = list(predictions.items())

            model_df = pd.DataFrame(
                {
                    "id": [k for k, _ in examples],
                    "output": [ex["prediction"] for _, ex in examples],
                    "pfer": [ex["pfer"] for _, ex in examples],
                    "fer": [ex["fer"] for _, ex in examples],
                    "per": [ex["per"] for _, ex in examples],
                    "fed": [ex["fed"] for _, ex in examples],
                }
            )

            project.upload_system(
                model_df, name=model_name, id_column="id", output_column="output"
            )


def main():
    args = parse_args()
    print(f"Pattern: {args.pattern}")
    data = load_data(args.pattern, args.num_samples, args.seed, args.sampled_items)
    print(f"Found {len(data)} datasets, {sum(len(m) for m in data.values())} models")
    upload_to_zeno(data, args.api_key, args.project_name)


if __name__ == "__main__":
    main()
