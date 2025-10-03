import yaml
import torch
import torchaudio
import kaldiio
from torch.utils.data import Dataset


class PowsmDataset(Dataset):
    def __init__(
        self, wav_scp_path, text_phoneme_path, language_path, sampling_rate=16000
    ):
        self.sampling_rate = sampling_rate
        self.wav_scp = self._load_wav_scp(wav_scp_path)
        self.text = self._load_text(text_phoneme_path)
        self.key2lang = self._extract_language(language_path)

        assert set(self.wav_scp.keys()).issubset(
            set(self.text.keys())
        ), "Extra key in wav.scp"
        self.keys = list(self.wav_scp.keys())
        # print(self.keys[:10], list(self.key2lang.keys())[:10])
        assert all(
            k in self.key2lang for k in self.keys
        ), "Missing language tags for some keys"
        print(
            f"Loaded dataset: {len(self.key2lang)} lang keys, {len(self.keys)} samples"
        )

    def _load_wav_scp(self, path):
        wav_scp = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key, wav_path = parts[0], parts[1]
                    if not wav_path.startswith("/work"):
                        wav_path = f"/work/hdd/bbjs/shared/powsm/s2t1/{wav_path}"
                    wav_scp[key] = wav_path
        return wav_scp

    def _load_text(self, path):
        text_dict = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    text_dict[parts[0]] = " ".join(
                        parts[1:]
                    )  # some examples have spaces in between we must retain full length of transcript
        return text_dict

    def _extract_language(self, path):
        key2lang = {}
        with open(path) as f:
            for line in f:
                key, tag = line.strip().split()[:2]
                if key.endswith("_pr"):
                    key = key[
                        :-3
                    ]  # remove _pr suffix for some datasets. This should be modified at source. It is bad design.
                key2lang[key] = tag.split("><")[0][1:].strip()
        return key2lang

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        wav_path = self.wav_scp[key]
        transcription = self.text[key]

        if ".ark" in wav_path:
            sr, wav = kaldiio.load_mat(wav_path)
            waveform = torch.from_numpy(wav).float().unsqueeze(0)
        else:
            waveform, sr = torchaudio.load(wav_path)

        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return {
            "key": key,
            "wav": waveform,
            "transcription": transcription,
            "wavpath": wav_path,
            "language": self.key2lang[key],
        }


def load_config(config_path="dataset_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_available_datasets(config_path="dataset_config.yaml"):
    return list(load_config(config_path)["datasets"].keys())


def get_inference_dataset(
    dataset_name, dataset_config_path="dataset_config.yaml", **kwargs
):
    config = load_config(dataset_config_path)

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_config = config["datasets"][dataset_name]
    for field in ["wav_scp", "text_phoneme", "language"]:
        if field not in ds_config:
            raise ValueError(f"Missing field '{field}' for dataset '{dataset_name}'")

    args = {
        "wav_scp_path": ds_config["wav_scp"],
        "text_phoneme_path": ds_config["text_phoneme"],
        "language_path": ds_config["language"],
        "sampling_rate": config.get("sampling_rate", 16000),
    }
    args.update(kwargs)
    return PowsmDataset(**args)


if __name__ == "__main__":
    print("Available datasets:", get_available_datasets())
    dataset = get_inference_dataset("buckeye")
    item = dataset[0]
    print(
        f"Sample: {item['key']}, shape: {item['wav'].shape}, text: {item['transcription']}"
    )
