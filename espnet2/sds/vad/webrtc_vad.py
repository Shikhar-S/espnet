import os

import librosa
import numpy as np
import torch
from typeguard import typechecked

from espnet2.sds.utils.utils import int2float
from espnet2.sds.vad.abs_vad import AbsVAD

try:
    import webrtcvad

    is_webrtcvad_available = True
except ImportError:
    is_webrtcvad_available = False


class WebrtcVADModel(AbsVAD):
    """Webrtc VAD Model"""

    @typechecked
    def __init__(
        self,
        speakup_threshold=12,
        continue_threshold=10,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        target_sr=16000,
    ):
        if not is_webrtcvad_available:
            raise ImportError("Error: webrtcvad is not properly installed.")
        super().__init__()
        self.vad_output = None
        self.vad_bin_output = None
        self.speakup_threshold = speakup_threshold
        self.continue_threshold = continue_threshold
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.target_sr = target_sr

    def warmup(self):
        return

    def forward(self, speech, sample_rate, binary=False):
        audio_int16 = np.frombuffer(speech, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        audio_float32 = librosa.resample(
            audio_float32, orig_sr=sample_rate, target_sr=self.target_sr
        )
        vad_count = 0
        for i in range(int(len(speech) / 960)):
            vad = webrtcvad.Vad()
            vad.set_mode(3)
            if vad.is_speech(speech[i * 960 : (i + 1) * 960].tobytes(), sample_rate):
                vad_count += 1
        if self.vad_output is None and vad_count > self.speakup_threshold:
            vad_curr = True
            self.vad_output = [torch.from_numpy(audio_float32)]
            self.vad_bin_output = [speech]
        elif self.vad_output is not None and vad_count > self.continue_threshold:
            vad_curr = True
            self.vad_output.append(torch.from_numpy(audio_float32))
            self.vad_bin_output.append(speech)
        else:
            vad_curr = False
        if self.vad_output is not None and vad_curr is False:
            array = torch.cat(self.vad_output).cpu().numpy()
            duration_ms = len(array) / self.target_sr * 1000
            if not (
                duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms
            ):
                if binary:
                    array = np.concatenate(self.vad_bin_output)
                self.vad_output = None
                self.vad_bin_output = None
                return array
        return None
