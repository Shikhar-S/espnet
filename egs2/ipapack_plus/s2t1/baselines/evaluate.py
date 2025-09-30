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

import os
import string
import unicodedata
import panphon.distance
from tqdm import tqdm
import json

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

# def get_metrics(hyps, refs):
#     ###############################################################
#     # formatting: make them phone sequence
#     # cleaner = {"ẽ": "ẽ", "ĩ": "ĩ", "õ": "õ", "ũ": "ũ", # nasal unicode
#     #             "ç": "ç", "g": "ɡ", # common unicode
#     #             "-": "", "'": "", " ":"", "͡": ""} # noise
#     # def clean(phones):
#     #     return "".join([cleaner.get(p, p) for p in phones])
#     removepunc = str.maketrans('', '', string.punctuation)
#     customized = {"g": "ɡ"}
#     def clean(sequence):
#         """
#         Normalize phones' unicode so that trie search can handle everything
#         Remove suprasegmental diacritics if specified
#         """
#         sequence = sequence.replace(" ", "")
#         sequence = unicodedata.normalize('NFD', sequence)
#         sequence = sequence.translate(removepunc)
#         sequence = ''.join([customized.get(c, c) for c in sequence])
#         return sequence
#     ###############################################################
#     hyps = [clean(x) for x in hyps]
#     refs = [clean(x) for x in refs]
#     # print(len(hyps), len(refs))
#     dst = panphon.distance.Distance()
#     result = {}
#     texts = zip(refs, hyps)
#     PFER = 0.0
#     for i, (ref, hyp) in enumerate(tqdm(texts, desc="Computing FER ZIPA style")):
#         PFER += dst.hamming_feature_edit_distance(hyp, ref)
#     PFER /= len(refs)
#     result['PFER'] = PFER
#     FED = dst.feature_edit_distance(hyps, refs)
#     FER = dst.feature_error_rate(hyps, refs ) * 100
#     PER = dst.phoneme_error_rate(hyps, refs) * 100
#     result.update({"FER": FER, "FED": FED, "PER": PER, "N": len(refs)})
#     return result

def get_metrics(hyps, refs):
    # Normalize inputs (remove spaces/punct, NFC->NFD, fix 'g'→'ɡ')
    removepunc = str.maketrans('', '', string.punctuation)
    customized = {"g": "ɡ"}
    def clean(s):
        s = s.replace(" ", "")
        s = unicodedata.normalize('NFD', s)
        s = s.translate(removepunc)
        return ''.join(customized.get(c, c) for c in s)

    hyps = [clean(x) for x in hyps]
    refs = [clean(x) for x in refs]
    N = len(refs)
    if N == 0:
        return {"PFER": 0.0, "FER": 0.0, "FED": 0.0, "PER": 0.0, "N": 0}

    dst = panphon.distance.Distance()

    # PFER: macro over utts, normalized by maxlen, scaled to percent
    pfer_sum = 0.0
    fed_sum = 0.0
    for h, r in tqdm(zip(hyps, refs), total=N, desc="Calculating metrics"):
        pfer_sum += dst.hamming_feature_edit_distance_div_maxlen(h, r) * 100.0
        fed_sum  += dst.feature_edit_distance(h, r)

    PFER = pfer_sum / N
    FED  = fed_sum

    # FER / PER: micro (batch helpers divide by total ref phones), scaled to %
    FER = dst.feature_error_rate(hyps, refs) * 100.0
    PER = dst.phoneme_error_rate(hyps, refs) * 100.0

    return {"PFER": PFER, "FER": FER, "FED": FED, "PER": PER, "N": N}

def compute_metrics(test_data):
    """Compute metrics"""
    hyps, refs =[], []
    for _, value in tqdm(test_data.items(), desc="Processing test data"):
        hyps.append(value['prediction'])
        refs.append(value['transcription'])
    metrics = get_metrics(hyps, refs)
    return metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Phoneme recognition evaluation')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--model', help='Model name for inference')
    parser.add_argument('--device', default='auto', help='Device to run inference on (e.g., cpu, cuda, auto)')
    parser.add_argument('--output_dir', default='./preds', help='Directory to save results')

    args = parser.parse_args()
    prediction_file = f"{args.output_dir}/{args.dataset}.{args.model.replace('/','.')}/preds.json"
    result_file = f"{args.output_dir}/{args.dataset}.{args.model.replace('/','.')}/metrics.json"
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    
    print(f"Loading: {prediction_file}")
    test_data = load_json(prediction_file)
    print(f"Loaded {len(test_data)} utterances from {prediction_file}")
    
    # Compute and display results
    metrics = compute_metrics(test_data)
    print(f"\n{args.model} on {args.dataset}")
    print("===================================")
    for k,v in metrics.items():
        print(f"{k}: {v:.2f}")
    print("===================================")
    save_json({**metrics, 'model': args.model, 'dataset': args.dataset}, result_file)

if __name__ == "__main__":
    main()