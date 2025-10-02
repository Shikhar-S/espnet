"""Run evaluation with baselines.
Usage:

INFERENCE:

For inference we need phonemizer and panphon 0.20. Run with:
    export PHONEMIZER_ESPEAK_LIBRARY=/work/nvme/bbjs/sbharadwaj/powsm/espeak-ng/installed/lib/libespeak-ng.so.1.1.51
    python baselines/run_inference.py \
        --dataset aishell \
        --model allophant 

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

import os
from tqdm import tqdm
import json
from egs2.ipapack_plus.s2t1.baselines.baselines_inference import get_inference_model
from egs2.ipapack_plus.s2t1.baselines.powsm_dataset import get_inference_dataset
import torch

def run_inference(dataset, dataset_config, model, device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = get_inference_dataset(dataset, dataset_config_path=dataset_config)
    model = get_inference_model(model, device=device)
    print(f"Running inference on {len(dataset)} utterances using {model.__class__.__name__} on {device}")

    test_data = {}
    for i,item in enumerate(tqdm(dataset)):
        pred = model.infer(item)
        test_data[item['key']] = {
            'key': item['key'], 
            'transcription': item['transcription'], 
            'prediction': pred,
            'wavpath': item['wavpath']
        }
        # if i>5: break
    return test_data

def load_json(results_file):
    """Load saved results from file"""
    with open(results_file, 'r') as f:
        json_data = json.load(f)
    return json_data

def save_json(data, out_file):
    """Save data to a json file"""
    with open(out_file, 'w') as f:
        json.dump(data, f , indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}")
    return

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Phoneme recognition inference')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--model', help='Model name for inference')
    parser.add_argument('--device', default='auto', help='Device to run inference on (e.g., cpu, cuda, auto)')
    parser.add_argument('--output_dir', default='./preds', help='Directory to save results')
    parser.add_argument('--dataset_config', default='/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/baselines/dataset_config.yaml', help='Path to dataset configuration file')

    args = parser.parse_args()
    prediction_file = f"{args.output_dir}/{args.dataset}.{args.model.replace('/','.')}/preds.json"
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    if os.path.exists(prediction_file):
        raise RuntimeError(f"Warning: {prediction_file} already exists!")
    print(f"Running: {args.model}")
    test_data = run_inference(args.dataset, args.dataset_config, args.model, args.device)
    save_json(test_data, prediction_file)

if __name__ == "__main__":
    main()