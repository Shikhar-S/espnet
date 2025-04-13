import argparse
import json
import random
import sys
import os

from glob import glob
from pathlib import Path
from typing import Optional, List, Dict

SPLIT_NAMES = ("train", "valid", "test")


def _get_args(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Prepare Nsynth dataset.")
    
    parser.add_argument("--root", type=Path, required=True, default="/data/user_data/hyejinsh/babel_user_data/nsynth/", help="The path of audio files")
    parser.add_argument("--task", type=str, required=True, default="instrument", help="Type of tasks - instrument or pitch")
    parser.add_argument("--output", type=Path, required=True, default="../data/", help="The path of output")
    
    args = parser.parse_args(argv)

    assert args.root.exists()

    return args


def _read_data(args, split):
    root_dir = os.path.normpath(f'{args.root}/nsynth-{split}')

    if not os.path.isdir(args.root):
        raise ValueError('The given root path is not a directory.'
                         f'\nI got {args.root}')

    json_path = os.path.join(root_dir, "examples.json")
    if not os.path.isfile(json_path):
        raise ValueError('The given root path does not contain an examples.json')

    print(f'Loading NSynth data from split {split} at {args.root}')

    with open(json_path, 'r') as fp:
        attrs = json.load(fp)

    # Dummy 'families' list to filter on; you should replace this based on your task.
    families = set(["keyboard", "guitar", "bass", "brass", "reed", "string", "synth_lead", "vocal", "organ", "mallet", "drums"])

    if args.task == "instrument":
        attrs = {k: a for k, a in attrs.items()
                 if a['instrument_family_str'] in families}

    if args.task == "pitch":
        attrs = {k: a for k, a in attrs.items()
                 if a['pitch'] in families}

    names = list(attrs.keys())

    return names, attrs


if __name__ == "__main__":
    print("Start preparing Nsynth dataset.")
    args = _get_args()
    print(args)

    args.output.mkdir(parents=True, exist_ok=True)

    files = {}
    for split in SPLIT_NAMES:
        (args.output / split).mkdir(exist_ok=True)
        files[split] = {
            "labels": open(args.output / split / "text", "w"),
            "path": open(args.output / split / "wav.scp", "w"),
            "dummy": open(args.output / split / "utt2spk", "w"),
        }

        names, attrs = _read_data(args, split)

        for index, (name, attr) in enumerate(zip(names, attrs)):
            key = f"nsynth-{split}-{index:05d}-{name}"
            path = f'{args.root}/audio/{name}.wav'
            label = "_".join(attr.split("_")[:-1])

            print(f"{key} {path}", file=files[split]["path"])
            #print(f"{key} {attrs[name]}", file=files[split]["labels"])
            print(f"{key} {label}", file=files[split]["labels"])
            print(f"{key} dummy", file=files[split]["dummy"])

    print("Done preparing Nsynth dataset.")
