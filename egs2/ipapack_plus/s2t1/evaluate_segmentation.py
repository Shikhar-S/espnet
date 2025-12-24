"""Usage:

python /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/evaluate_segmentation.py \
    --pred /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/preds/test_buckeye/pr-trial1-0.jsonl \
    --gt /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/aligned.cuts.000000.jsonl \
    -o results_clean.json

NOTE(shikhar): This scripts breaks the ground truth phonemes into single character phonemes and divides the
    time span among phoneme's characters.
    This is because the annotations in buckeye and the training data are not a 1-1 match. 
    For example, a single phoneme with multiple characters breaks into more than one tokens. 
    Ways to handle this better in future:
    [Step 2] Combine the predictions based on the ground truth phoneme sequence.

    However for now this is tricky because ground truth is constructed from the orthography field in jsonl files. 
    Orthography is a noisy field (some words are missing) and leads to alignment issues when combining the characters.
    Ways to handle this better in future:
    [Step 1] Use the phoneme field in jsonl files which match the training/test targets. However note that 
    sometimes the phoneme is missing from the vocabulary which can be an edge case to handle. Usually it shows up 
    as <unk> in the predictions.
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
from tabulate import tabulate


def load_jsonl(file_path):
    """Load JSONL file indexed by ID."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data[entry['id']] = entry
    return data


def format_gt_alignment_list(alignment):
    """Return (phoneme, start, end, is_missing) tuples. Break all paired phonemes 
    like the predicted string. This is a difference from the standard data preparation."""
    result = []
    if len(alignment) == 0:
        return result
    start_time=0
    start_time = alignment[0][1]
    for phoneme_align in alignment:
        phoneme, start,end = phoneme_align
        start = (start - start_time) * 1000
        end = (end - start_time) * 1000
        valid_gt = False
        if start > end:
            valid_gt = True
        phlen = len(phoneme)
        total_len=float(end)-float(start)
        for j,ph in enumerate(phoneme):
            result.append((ph, float(start + j*total_len/phlen), float(start + (j+1)*total_len/phlen), valid_gt))
    return result

def format_pred_alignment_list(alignment, gt_alignment):
    """Align the predicted alignment to the ground truth alignment."""
    alignment=alignment[1:] # skip first blank
    alignment = [(ph, (start,end)) for ph, (start,end) in alignment if ph!='<unk>' and len(ph)>0]
    # Ideally there should be no <unk> and empty phonemes in predictions as we feed the model with ground truth transcripts from vocabulary.
    gtidx=0
    predidx=0
    result=[]
    while gtidx < len(gt_alignment) and predidx < len(alignment):
        gt_char = gt_alignment[gtidx][0].strip()
        pred_char = alignment[predidx][0].strip()
        if pred_char == gt_char:
            result.append((pred_char, float((alignment[predidx][1][0])), float((alignment[predidx][1][1])), False))
            predidx+=1
            gtidx+=1
        elif len(pred_char) > len(gt_char) and pred_char.startswith(gt_char):
            predchrlen=len(pred_char)
            start=float((alignment[predidx][1][0]))
            end=float((alignment[predidx][1][1]))
            predduration = end-start
            result.append((gt_char, start, start+predduration/predchrlen, False))
            pred_char = pred_char[len(gt_char):]
            alignment[predidx] = (pred_char, (start+predduration/predchrlen, end))
            gtidx+=1
        else:
            # since gt_char is always len==1, this means pred_char is different or 
            # ground truth has missing phonemes. We skip such pred_chars
            predidx+=1
    return result

def compute_boundary_metrics(pred_s, pred_e, gt_s, gt_e):
    start_err = abs(pred_s - gt_s)
    end_err = abs(pred_e - gt_e)
    phoneme_pbe = 0.5 * (start_err + end_err)
    gt_phoneme_duration = gt_e - gt_s
    pred_phoneme_duration = pred_e - pred_s
    return {
        'start_err': start_err,
        'end_err': end_err,
        'phoneme_pbe': phoneme_pbe,
        'gt_phoneme_duration': gt_phoneme_duration,
        'pred_phoneme_duration': pred_phoneme_duration
    }


def compute_pbe(pred_file, gt_file):
    """Compute PBE between prediction and ground truth files."""
    preds = load_jsonl(pred_file)
    gts = load_jsonl(gt_file)
    common_ids = set(preds.keys()) & set(gts.keys())
    
    total_phonemes = used_phonemes = missing_phonemes = 0
    all_start_errors, all_end_errors, utterance_pbes = [], [], []
    duration_pred =[]
    duration_gt = []
    c=0
    for entry_id in tqdm(sorted(common_ids), desc="Processing entries"):
        # if entry_id!='s40_9824-9070': continue
        # if entry_id!='s40_9824-9070': continue
        # if entry_id!='s38_8928-5777': continue
        # if entry_id!='s22_8222-5653': continue

        # if entry_id!='s18_9445-646': continue
        # print('=='*20)
        # print(preds[entry_id]['alignment'])
        # print('--'*20)
        # print(gts[entry_id]['alignment'])
        # print('=='*20)
        # print(''.join([ph for ph,_ in preds[entry_id]['alignment']]).strip())
        # print(''.join([ph for ph,_,_ in gts[entry_id]['alignment']]).strip())
        # print('=='*20)
        c+=1
        gt_align = format_gt_alignment_list(gts[entry_id]['alignment'])
        pred_align = format_pred_alignment_list(preds[entry_id]['alignment'], gt_align)
        pred_str = ''.join([ph for ph,_,_,_ in pred_align]).strip()
        gt_str = ''.join([ph for ph,_,_,_ in gt_align]).strip()
        if pred_str!=gt_str:
            if not gt_str.startswith(pred_str):
                # Sometimes the ground truth has extra phonemes at the end
                # print(f"GT longer than pred for {entry_id} -- pred {pred_str} vs gt {gt_str}")
                # continue
                print(f"Length mismatch for {entry_id} -- pred {pred_str} vs gt {gt_str}")
            # continue
        phoneme_pbes = []
        for (pred_ph, pred_s, pred_e, pred_miss), (gt_ph, gt_s, gt_e, gt_miss) in zip(pred_align, gt_align):
            # print(pred_ph, '--', gt_ph)
            total_phonemes += 1
            if gt_miss or pred_miss:
                # print(f"Missing phoneme for {entry_id}: pred ({pred_ph}, {pred_s}, {pred_e}, {pred_miss}) vs gt ({gt_ph}, {gt_s}, {gt_e}, {gt_miss})")
                # print(pred_str, '---', gt_str)
                missing_phonemes += 1
            else:
                assert not pred_miss, f"Pred missing but GT present for {entry_id}"
                assert pred_ph == gt_ph, f"Phoneme mismatch for {entry_id}: pred {pred_ph} vs gt {gt_ph}"


                metrics = compute_boundary_metrics(pred_s, pred_e, gt_s, gt_e)
                start_err = metrics['start_err']
                end_err = metrics['end_err']
                phoneme_pbe = metrics['phoneme_pbe']
                gt_phoneme_duration = metrics['gt_phoneme_duration']
                pred_phoneme_duration = metrics['pred_phoneme_duration']

                all_start_errors.append(start_err)
                all_end_errors.append(end_err)
                phoneme_pbes.append(phoneme_pbe)
                duration_gt.append(gt_phoneme_duration)
                duration_pred.append(pred_phoneme_duration)

                used_phonemes += 1
        
        if phoneme_pbes:
            utterance_pbes.append(np.mean(phoneme_pbes))
    
    return {
        'entries': len(common_ids),
        'total_phonemes': total_phonemes,
        'used_phonemes': used_phonemes,
        'missing_phonemes': missing_phonemes,
        'pbe_ms': np.mean(utterance_pbes),
        'pbe_std_ms': np.std(utterance_pbes),
        'pbe_median_ms': np.median(utterance_pbes),
        'mean_start_error_ms': np.mean(all_start_errors),
        'mean_end_error_ms': np.mean(all_end_errors),
        'mean_boundary_error_ms': np.mean(all_start_errors + all_end_errors),
        'mean_gt_duration_ms': np.mean(duration_gt),
        'mean_pred_duration_ms': np.mean(duration_pred),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute PBE between pred and GT alignments')
    parser.add_argument('--predictions', '--pred', required=True, help='Predictions JSONL file')
    parser.add_argument('--ground-truth', '--gt', required=True, help='Ground truth JSONL file')
    parser.add_argument('--output', '-o', help='Save results to JSON')
    
    args = parser.parse_args()
    
    results = compute_pbe(args.predictions, getattr(args, 'ground_truth'))
    
    table_data = [
        ["Entries", results['entries']],
        ["Phonemes (used/total)", f"{results['used_phonemes']}/{results['total_phonemes']} ({results['used_phonemes']/results['total_phonemes']:.1%})"],
        ["PBE (ms)", f"{results['pbe_ms']:.3f} ± {results['pbe_std_ms']:.3f}"],
        ["Start error (ms)", f"{results['mean_start_error_ms']:.3f}"],
        ["End error (ms)", f"{results['mean_end_error_ms']:.3f}"],
        ["Boundary error (ms)", f"{results['mean_boundary_error_ms']:.3f}"],
        ["PBE Median (ms)", f"{results['pbe_median_ms']:.3f}"],
        ["Mean GT duration (ms)", f"{results['mean_gt_duration_ms']:.3f}"],
        ["Mean Pred duration (ms)", f"{results['mean_pred_duration_ms']:.3f}"],
    ]

    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="plain"))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()