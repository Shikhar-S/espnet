"""Prepares data for Places Hindi dataset."""

import os
import sys
import json
import logging

if __name__ == "__main__":
    ROOT_DATA_DIR = sys.argv[1]
    SPLIT_NAMES = ["dev", "test", "train"]

    for data_split in SPLIT_NAMES:
        metadata_file = os.path.join(ROOT_DATA_DIR, f"PlacesHindi100k_{data_split}.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)['data']
        
        text_file_path = os.path.join("data",data_split,"text")
        wav_file_path = os.path.join("data",data_split,"wav.scp")
        utt2spk_file_path = os.path.join("data",data_split,"utt2spk")
        missing_text_keys=0
        missing_wav_files=0
        with open(text_file_path,'w') as text_f, open(wav_file_path,'w') as wav_scp_f, open(utt2spk_file_path,'w') as utt2spk_f:
            for uttid, data in enumerate(metadata):
                try:
                    text = data['asr_text']
                    wavpath = data['wav']
                    wavpath = os.path.join(ROOT_DATA_DIR, wavpath)
                    if not os.path.exists(wavpath):
                        missing_wav_files+=1
                        continue
                    print(f"{uttid} {text}", file=text_f)
                    print(f"{uttid} {wavpath}", file=wav_scp_f)
                    print(f"{uttid} dummy", file=utt2spk_f)
                except KeyError:
                    missing_text_keys+=1
                    logging.warning(f"Missing keys for {data}")
        print(f"Missing text keys: {missing_text_keys}, Missing wav files: {missing_wav_files}")