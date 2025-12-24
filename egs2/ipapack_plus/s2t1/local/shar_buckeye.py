import os
import lhotse
from lhotse import CutSet, Mfcc, Fbank
import webdataset as wds
from glob import glob
from tqdm import tqdm
import json
from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download, safe_extract
from pathlib import Path
import logging
import pandas as pd


logging.basicConfig(level=logging.DEBUG,  # Set the logging level to DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')



def parse_utterance(
    phones,
    text,
    filename,
    audio_path,
    lang
): 
    
    #ind = row[0]
    recording_id = filename
    
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    
    
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=lang,
        speaker=None,
        text=phones,
        alignment=None,
        custom={'orthographic': text}
    )
    return recording, segment



if __name__ == "__main__":
    
    path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/buckeye_split'
    

    lang = 'eng'


    audio_path = Path(path)

    supervisions = []
    recordings = []

    for dirpath, _, filenames in os.walk(path):
        for filename in tqdm(filenames):
            if filename.endswith('.wav'):  # Check if the file is a .phn file
                audio_path = os.path.join(dirpath, filename)
                with open(audio_path.replace('.wav','.ipa'),'r') as f:
                    ipa = f.read()
                text = ''
                with open(audio_path.replace('.wav','.word'),'r') as file:
                    for line in file.readlines():
                        _, _, w = line.split()
                        text = text+w+' '
                        
                result = parse_utterance(ipa, text, filename.replace('.wav',''), audio_path, lang)
                if result is None:
                    continue
                recording,supervision = result
                supervisions.append(supervision)
                recordings.append(recording)

    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set = RecordingSet.from_recordings(recordings)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    manifests = {"recordings": recording_set, "supervisions": supervision_set}

    cuts = CutSet.from_manifests(recordings=recording_set,supervisions=supervision_set)
    resampled_cuts = cuts.resample(16000)

    data_dir = Path('/scratch/lingjzhu_root/lingjzhu1/lingjzhu/multilingual/buckeye_shar/')
    data_dir.mkdir(parents=True, exist_ok=True)
    shards = resampled_cuts.to_shar(data_dir, fields={"recording": "flac"}, shard_size=60000)
    print(shards)