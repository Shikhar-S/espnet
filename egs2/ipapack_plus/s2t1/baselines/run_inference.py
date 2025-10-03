"""Run evaluation with baselines.
Usage:

INFERENCE:

For inference we need phonemizer and panphon 0.20. Run with:
    export PHONEMIZER_ESPEAK_LIBRARY=/work/nvme/bbjs/sbharadwaj/powsm/espeak-ng/installed/lib/libespeak-ng.so.1.1.51
    python baselines/run_inference.py \
        --dataset buckeye \
        --model powsm \
        --num_workers 1 \
        --s2t_train_config /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/config.yaml \
        --s2t_model_file /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/valid.acc.ave_5best.till40epoch.pth \
        --bpemodel /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/data/token_list/bpe_unigram40000/bpe.model \
        --beam_size 1 \
        --ctc_weight 0.3 

        For powsm model we only check for "powsm" as first few chars of model name, 
        you can name your custom checkpoint and inference condition "powsm_arg1val1_arg2val2" 
        (do not use dots in name).

Available models:
'powsm' \
'facebook/wav2vec2-lv-60-espeak-cv-ft' 'facebook/wav2vec2-xlsr-53-espeak-cv-ft' \
'ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns' 'allophant' 'allosaurus'

Mem sizes for inference (1 example):
'facebook/wav2vec2-lv-60-espeak-cv-ft' 2.4G
powsm 4.7G

Available datasets:
'aishell' 'buckeye' 'cv' 'doreco' 'epadb' 'fleurs' \
'fleurs_indv' 'kazakh' 'l2arctic' 'librispeech' \
'mls_dutch' 'mls_french' 'mls_german' \
'mls_italian' 'mls_polish' 'mls_portuguese' \
'mls_spanish' 'southengland' 'speechoceannotth' \
'tamil' 'tusom2021' 'voxangeles'
"""

import os
from tqdm import tqdm
import json
from egs2.ipapack_plus.s2t1.baselines.inference_modules import get_inference_model
from egs2.ipapack_plus.s2t1.baselines.powsm_dataset import get_inference_dataset
import torch
import multiprocessing as mp
from functools import partial

import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def _work_chunk(dataset, idxs, model_name, device, kwargs):
    model = get_inference_model(model_name, device=device, **kwargs)
    out = {}
    for i in tqdm(idxs, desc="Processing", leave=False):
        it = dataset[i]
        pred = model.infer(it)
        out[it["key"]] = {
            "key": it["key"],
            "transcription": it["transcription"],
            "prediction": pred,
            "wavpath": it["wavpath"],
        }
    return out


def run_inference(
    dataset, dataset_config, model, device, num_workers=1, **model_kwargs
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = get_inference_dataset(dataset, dataset_config_path=dataset_config)
    # ds = [ds[kk] for kk in range(10)]  # DEBUGGING
    print(
        f"Running inference on {len(ds)} utterances using {model} on {device} with {num_workers} workers"
    )

    N = len(ds)
    if num_workers <= 1:
        return _work_chunk(ds, range(N), model, device, model_kwargs)

    cs = (N + num_workers - 1) // num_workers
    chunks = [
        range(i * cs, min((i + 1) * cs, N)) for i in range(num_workers) if i * cs < N
    ]
    worker = partial(
        _work_chunk, ds, model_name=model, device=device, kwargs=model_kwargs
    )

    with mp.get_context("spawn").Pool(num_workers) as pool:
        parts = list(tqdm(pool.imap(worker, chunks), total=len(chunks), desc="Chunks"))
    out = {}
    for p in parts:
        out.update(p)
    return out


def load_json(results_file):
    """Load saved results from file"""
    with open(results_file, "r") as f:
        json_data = json.load(f)
    return json_data


def save_json(data, out_file):
    """Save data to a json file"""
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}")
    return


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phoneme recognition inference")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--model", help="Model name for inference")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on (e.g., cpu, cuda, auto)",
    )
    parser.add_argument(
        "--output_dir", default="./preds", help="Directory to save results"
    )
    parser.add_argument(
        "--dataset_config",
        default="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/baselines/dataset_config.yaml",
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of parallel workers for inference",
    )
    #### powsm specific args ####
    parser.add_argument("--s2t_train_config", help="Path to s2t_train config")
    parser.add_argument("--s2t_model_file", help="Path to s2t model file")
    parser.add_argument("--bpemodel", help="Path to bpe model")
    parser.add_argument("--beam_size", default=1, type=int, help="Beam size")
    parser.add_argument("--ctc_weight", default=0.3, type=float, help="CTC weight")
    parser.add_argument(
        "--text_prev", default="<na>", type=str, help="Previous text input"
    )
    parser.add_argument("--lang_sym", default="<unk>", type=str, help="Language symbol")
    parser.add_argument("--task_sym", default="<pr>", type=str, help="Task symbol")
    parser.add_argument(
        "--return_ctc",
        action="store_true",
        help="Whether to return CTC states instead of transcription",
    )
    #### powsm specific args end ####

    args = parser.parse_args()
    prediction_file = (
        f"{args.output_dir}/{args.dataset}.{args.model.replace('/','.')}/preds.json"
    )
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    if os.path.exists(prediction_file):
        raise RuntimeError(f"Warning: {prediction_file} already exists!")

    # Prepare model-specific kwargs
    model_kwargs = {}
    if args.model == "powsm":
        model_kwargs = {
            "s2t_train_config": args.s2t_train_config,
            "s2t_model_file": args.s2t_model_file,
            "bpemodel": args.bpemodel,
            "beam_size": args.beam_size,
            "ctc_weight": args.ctc_weight,
            "text_prev": args.text_prev,
            "lang_sym": args.lang_sym,
            "task_sym": args.task_sym,
            "return_ctc": args.return_ctc,
        }

    print(f"Running: {args.model}")
    test_data = run_inference(
        args.dataset,
        args.dataset_config,
        args.model,
        args.device,
        num_workers=args.num_workers,
        **model_kwargs,
    )
    save_json(test_data, prediction_file)


if __name__ == "__main__":
    main()
