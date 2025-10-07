"""
Average multiple PyTorch checkpoints.
Usage:
    python average_checkpoints.py \
        --ckpts exp_dtai_1task/s2t_train_s2t_transformer_deltaai_100m_raw_bpe40000_mega/41epoch.pth \
        --ckpts exp_dtai_1task/s2t_train_s2t_transformer_deltaai_100m_raw_bpe40000_mega/42epoch.pth \
        --ckpts exp_dtai_1task/s2t_train_s2t_transformer_deltaai_100m_raw_bpe40000_mega/43epoch.pth \
        --ckpts exp_dtai_1task/s2t_train_s2t_transformer_deltaai_100m_raw_bpe40000_mega/44epoch.pth \
        --ckpts exp_dtai_1task/s2t_train_s2t_transformer_deltaai_100m_raw_bpe40000_mega/45epoch.pth \
        --out exp_dtai_1task/s2t_train_s2t_transformer_deltaai_100m_raw_bpe40000_mega/valid.acc.ave_5best.till45epoch.pth
"""

import torch, argparse


def average_checkpoints(paths):
    avg = None
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        if avg is None:
            avg = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg:
                avg[k] += state[k].float()
    for k in avg:
        avg[k] /= len(paths)
    return avg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Average multiple PyTorch checkpoints.")
    ap.add_argument(
        "--ckpts", nargs="+", required=True, help="List of checkpoint paths"
    )
    ap.add_argument("--out", required=True, help="Output path for averaged checkpoint")
    args = ap.parse_args()

    avg_state = average_checkpoints(args.ckpts)
    torch.save(avg_state, args.out)
    print(f"Averaged checkpoint saved to {args.out}")
