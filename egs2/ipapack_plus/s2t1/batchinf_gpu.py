#!/usr/bin/env python3

"""
Batch inference script for speech-to-text pronunciation recognition.
s14_5166
# WAVS AT: /work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/recording
# METADATA AT: /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_buckeye/cuts.000000.jsonl
# English transcript AT: /work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/text.asr
# Phonetic transcript AT: /work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/text.raw

Example usage:
    python batchinf_gpu.py \
        --testset test_buckeye \
        --taskname trial1 \
        --basedir /work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1 \
        --expdir exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000 \
        --split 0 \
        --device gpu
"""
import argparse
import os
import torch
import torchaudio
from espnet2.bin.s2t_inference import Speech2Text
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from espnet2.torch_utils.device_funcs import to_device
import matplotlib.pyplot as plt
import json

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch inference for speech-to-text pronunciation recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="Name of test set (e.g., test_buckeye, test_l2artic)"
    )
    
    parser.add_argument(
        "--taskname", 
        type=str,
        required=True,
        help="Name of this setting/task"
    )
    
    parser.add_argument(
        "--basedir",
        type=str, 
        required=True,
        help="Path to exp_name/conf_name directory"
    )
    
    parser.add_argument(
        "--expdir",
        type=str,
        required=True, 
        help="Experiment directory path (expdir/expname)"
    )
    
    parser.add_argument(
        "--split",
        type=int,
        required=True,
        choices=range(16),
        help="Split number (0-15)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run inference on (cpu or gpu). Default: cpu"
    )
    
    return parser.parse_args()


def align(speech, text, inference_engine, device):
    # print(text)
    # print(type(speech))
    # Prepare speech
    # TODO(shikhar): the text here contains starting tags and misses sos etc
    POWSM_TIME_HOP=0.02
    if isinstance(speech, np.ndarray):
        speech = torch.tensor(speech)

    # Only support single-channel speech
    if speech.dim() > 1:
        assert (
            speech.dim() == 2 and speech.size(1) == 1
        ), f"speech of size {speech.size()} is not supported"
        speech = speech.squeeze(1)  # (nsamples, 1) --> (nsamples,)

    speech_length = int(
        inference_engine.preprocessor_conf["fs"] * inference_engine.preprocessor_conf["speech_length"]
    )
    original_speech_length = speech.size(-1)
    # Pad or trim speech to the fixed length
    if original_speech_length >= speech_length:
        speech = speech[:speech_length]
    else:
        speech = F.pad(speech, (0, speech_length - original_speech_length))

    # Batchify input
    # speech: (nsamples,) -> (1, nsamples)
    speech = speech.unsqueeze(0).to(getattr(torch, inference_engine.dtype))
    # lengths: (1,)
    speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.shape[1])

    # Prepare text
    text = inference_engine.converter.tokens2ids(inference_engine.tokenizer.text2tokens(text))
    text = torch.tensor([text], device=device)
    text_lengths = text.new_full([1], dtype=torch.long, fill_value=text.shape[1])
    batch={'speech': speech, 'speech_lengths': speech_lengths, 'text': text, 'text_lengths': text_lengths}
    
    batch = to_device(batch, device)
    ########## ALIGN ##########
    align_label, align_score = inference_engine.s2t_model.forced_align(**batch) # (1, 998)
    align_label_spans = torchaudio.functional.merge_tokens(align_label[0], align_score[0])
    
    # print('==='*20)
    ret=[]
    for span in align_label_spans:
        text_token = inference_engine.converter.ids2tokens([span.token])[0]
        start_time = span.start * POWSM_TIME_HOP
        end_time = span.end * POWSM_TIME_HOP
        ret.append((text_token.replace('/',''), [start_time * 1000, end_time * 1000]))
        # print(f"{text_token}\t{start_time:.2f}\t{end_time:.2f}\t{span.score:.4f}")
    # print('==='*20)

    # word_spans2 = unflatten(align_label_spans, [len(tokenizer.encode_flatten(word)) for word in transcript.split()])
    # original_feature_length = original_speech_length // 320
    # print(original_feature_length)
    # print(align_score.shape)
    # print(align_label.shape)
    # print(align_label[0, :original_feature_length])
    # print(text)
    # print(align_label_spans)
    return align_label_spans, ret

def plot_spectrogram_with_tokens(speech, align_label_spans, inference_engine, filename, sample_rate=16000):
    """
    Plot spectrogram with token alignments overlaid and save to file.
    
    Args:
        speech: audio waveform (numpy array or tensor)
        align_label_spans: token spans from alignment
        inference_engine: Speech2Text model for token conversion
        filename: name for saving the plot
        sample_rate: audio sample rate
    """
    
    # Convert to numpy if tensor
    if isinstance(speech, torch.Tensor):
        speech = speech.numpy()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot waveform
    time_axis = np.linspace(0, len(speech) / sample_rate, len(speech))
    ax1.plot(time_axis, speech)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Waveform with Token Alignments')
    ax1.grid(True, alpha=0.3)
    
    # Plot spectrogram
    ax2.specgram(speech, Fs=sample_rate, NFFT=512, noverlap=256, cmap='viridis')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Spectrogram with Token Alignments')
    
    # Add token overlays
    colors = plt.cm.Set3(np.linspace(0, 1, len(align_label_spans)))
    
    for i, span in enumerate(align_label_spans):
        # Convert token ID to text
        text_token = inference_engine.converter.ids2tokens([span.token])[0]
        start_time = span.start * 0.02  # Convert frame to time
        end_time = span.end * 0.02
        
        # Skip empty tokens or special tokens
        if text_token in ['<blank>', '<sos>', '<eos>', '▁']:
            continue
            
        color = colors[i % len(colors)]
        
        # Add vertical lines and shaded regions
        for ax in [ax1, ax2]:
            ax.axvspan(start_time, end_time, alpha=0.3, color=color, edgecolor='black', linewidth=1)
            
        # Add token labels
        mid_time = (start_time + end_time) / 2
        ax1.text(mid_time, ax1.get_ylim()[1] * 0.9, text_token, 
                ha='center', va='bottom', fontsize=8, rotation=0,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        ax2.text(mid_time, ax2.get_ylim()[1] * 0.9, text_token,
                ha='center', va='bottom', fontsize=8, rotation=0,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f"plots/{filename}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Saved plot to: {output_path}")


def main():
    TOTAL_SPLITS = 1
    args = parse_arguments()
    
    # Determine language based on testset
    if args.testset in ["test_buckeye", "test_l2artic"]:
        lang = "<eng>"
    else:
        lang = "<unk>"

    readonly_expdir = f"{args.basedir}/{args.expdir}"
    
    # Determine device
    if args.device == "gpu" and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        if args.device == "gpu" and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA not available. Using CPU instead.")
        print("Using CPU")

    bpebase = "bpe_unigram40000"
    model = Speech2Text(
        s2t_train_config=f"{args.expdir}/config.yaml",
        s2t_model_file=f"{readonly_expdir}/valid.acc.ave_5best.till40epoch.pth",
        bpemodel=f"{args.basedir}/data/token_list/{bpebase}/bpe.model", 
        beam_size=1, 
        ctc_weight=0.3,
        device=device
    )

    # Load audio file paths and names
    with open(f"dump/raw/{args.testset}/wav.scp", "r") as f:
        audios = [line.split()[1] for line in f.readlines()]
    with open(f"dump/raw/{args.testset}/text", "r") as f:
        names_and_texts = [line.split(maxsplit=1) for line in f.readlines()]
        names = [nt[0] for nt in names_and_texts]
        texts = [nt[1].split(maxsplit=1)[1] if len(nt) > 1 else "" for nt in names_and_texts] # remove the context info

    # Create 16 splits
    size = len(audios)
    start = args.split * (size // TOTAL_SPLITS)
    end = (args.split + 1) * (size // TOTAL_SPLITS)
    if args.split == TOTAL_SPLITS-1:  # last split takes the rest
        end = size

    # Check for existing predictions to resume from
    pred_file = f"preds/{args.testset}/pr-{args.taskname}-{args.split}.jsonl"
    if os.path.exists(pred_file):
        with open(pred_file, "r") as f:
            skip = sum(1 for _ in f)
    else:
        skip = 0
        # Create directory if it doesn't exist
        os.makedirs(f"preds/{args.testset}", exist_ok=True)
    
    skip += start

    print(f"Processing split {args.split}: samples {start} to {end-1}")
    print(f"Skipping first {skip - start} samples (already processed)")
    os.makedirs("plots", exist_ok=True)
    print("Plots will be saved to the 'plots' directory.")

    # Process audio files
    for i in tqdm(range(start, end), desc=f"Split {args.split}"):
        if i < skip:
            continue
            
        speech, _ = sf.read(audios[i])
        text = texts[i]
        
        na = "<na>"
        # print('Aligning')
        alignment, phoneme2ts = align(speech, text, model, device)
        # plot_filename = f"{args.testset}_{args.taskname}_{names[i]}"
        # plot_spectrogram_with_tokens(speech, alignment, model, plot_filename, sample_rate=16000)

        write_dict = {'id': names[i], 'alignment': phoneme2ts}

        # if i>10:
            # break
        with open(pred_file, "a") as prpred:
            prpred.write(json.dumps(write_dict, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()