"""Prepares data for WavCaps dataset."""

import soundfile as sf
import os
import sys
import json

SKIP_THRESHOLD_SECONDS = 30  # Some files are too long, so we skip them.
FRAME_RATE = 32_000

if __name__ == "__main__":
    ROOT_DATA_DIR = sys.argv[1]
    DATASET_COMPONENTS = {
        "AudioSet_SL": "as_final.json",
        "BBC_Sound_Effects": "bbc_final.json",
        "FreeSound": "fsd_final.json",
        "SoundBible": "sb_final.json",
    }

    # These are the audio clips that are present in AudioCap, and should be excluded because we combine audiocap later separately.
    BLACKLIST = "blacklist_exclude_all_ac.json"
    JSON_DIR = "json_files"
    # Extracted flac files reside here.
    AUDIO_DIR_PATTERN = "Zip_files/{dataset}/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/{dataset}_flac"

    wav_scp_write_path = os.path.join("data", "wavcaps_train", "wav.scp")
    text_write_path = os.path.join("data", "wavcaps_train", "text")
    # Create directory and overwrite if it exists.
    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)
    os.makedirs(os.path.dirname(text_write_path), exist_ok=True)
    n_total = 0
    with open(wav_scp_write_path, "w", encoding="utf - 8") as wav_scp_f, open(
        text_write_path, "w", encoding="utf - 8"
    ) as text_f:
        for dataset, json_name in DATASET_COMPONENTS.items():
            print(f"Processing {dataset}...")
            n_multichannel = 0
            n_skipped = 0
            n_processed = 0
            n_cropped = 0
            with open(
                os.path.join(ROOT_DATA_DIR, JSON_DIR, dataset, json_name),
                "r",
            ) as f:
                json_d = json.load(f)
            for data in json_d["data"]:
                file_id = data["id"]
                if dataset == "AudioSet_SL" and file_id.endswith(
                    ".wav"
                ):  # For some reason AudioSet_SL has .wav extension (but soxi gives flac), so we correct it
                    file_id = file_id[:-4]
                uttid = f"{dataset}_{file_id}"
                text = data["caption"]
                audio_path = os.path.join(
                    ROOT_DATA_DIR,
                    AUDIO_DIR_PATTERN.format(dataset=dataset),
                    f"{file_id}.flac",
                )
                if not os.path.exists(audio_path):
                    print(f"Skipping {uttid} as {audio_path} does not exist.")
                    n_skipped += 1
                    continue
                # Check audio clip duration.
                if data["duration"] > SKIP_THRESHOLD_SECONDS:
                    audio, sr = sf.read(
                        audio_path, start=0, frames=SKIP_THRESHOLD_SECONDS * FRAME_RATE
                    )
                    audio_path = os.path.join(
                        "local",
                        "copied_data",
                        f"{file_id}_0_{SKIP_THRESHOLD_SECONDS}sec.flac",
                    )
                    if len(audio.shape) > 1:
                        # Some files are multichannel, so we skip them.
                        n_multichannel += 1
                        n_processed += 1
                        continue
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    sf.write(audio_path, audio, sr)
                    n_cropped += 1
                    n_processed += 1
                if data["duration"] <= 0.1:
                    n_skipped += 1
                    # Some files are too short, so we skip them. Some even have 0 second duration!!
                    continue
                print(f"{uttid} {audio_path}", file=wav_scp_f)
                print(f"{uttid} {text}", file=text_f)
                n_processed += 1
                if (n_processed) % 10 == 0:
                    print(
                        f"Processed {n_processed} files. Skipped {n_skipped} files, cropped {n_cropped} and {n_multichannel} were multi-channel."
                    )
                    break
            print(
                f"Processed {n_processed} files and skipped {n_skipped} files, cropped {n_cropped} and {n_multichannel} were multi-channel from {dataset}."
            )
            n_total += n_processed
        print(f"Processed {n_total} files in total.")
