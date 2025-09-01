"""Prepares data for Places Hindi dataset."""

import os
import sys
import json
import logging
import soundfile as sf

MAX_DURATION = 30.0

if __name__ == "__main__":
    ROOT_DATA_DIR = sys.argv[1]
    SPLIT_NAMES = ["dev", "test", "train"]

    for data_split in SPLIT_NAMES:
        metadata_file = os.path.join(ROOT_DATA_DIR, f"PlacesHindi100k_{data_split}.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)['data']
        print(len(metadata), 'files found in', data_split)
        text_file_path = os.path.join("data",data_split,"text")
        wav_file_path = os.path.join("data",data_split,"wav.scp")
        utt2spk_file_path = os.path.join("data",data_split,"utt2spk")
        missing_text_keys=0
        missing_wav_files=0
        long_files=0
        with open(text_file_path,'w') as text_f, open(wav_file_path,'w') as wav_scp_f, open(utt2spk_file_path,'w') as utt2spk_f:
            for uttid, data in enumerate(metadata):
                try:
                    text = data['asr_text']
                    wavpath = data['wav']
                    wavpath = os.path.join(ROOT_DATA_DIR, wavpath)
                    if not os.path.exists(wavpath):
                        missing_wav_files+=1
                        continue
                    if data_split=='train' or data_split=='dev':
                        # filter files that are too long
                        duration = sf.info(wavpath).duration
                        if duration > MAX_DURATION:
                            long_files+=1
                            continue
                    print(f"{uttid} {text}", file=text_f)
                    print(f"{uttid} {wavpath}", file=wav_scp_f)
                    print(f"{uttid} dummy", file=utt2spk_f)
                    if (uttid+1) % 1000==0:
                        print(f'Processed {uttid+1} files.')
                except KeyError:
                    missing_text_keys+=1
                    logging.warning(f"Missing keys for {data}")
        print(f"Missing text keys: {missing_text_keys}, Missing wav files: {missing_wav_files}. Long files: {long_files}")