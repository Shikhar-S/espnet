import torch
import soundfile as sf
from espnet2.bin.s2t_inference import Speech2Text


class PowsmInference:
    def __init__(self, device="cpu", **kwargs):
        self.device = device
        self.s2t_train_config = kwargs.get("s2t_train_config")
        self.s2t_model_file = kwargs.get("s2t_model_file")
        self.bpemodel = kwargs.get("bpemodel")
        assert (
            self.s2t_train_config is not None
        ), "s2t_train_config path must be provided"
        assert self.s2t_model_file is not None, "s2t_model_file path must be provided"
        assert self.bpemodel is not None, "bpemodel path must be provided"

        beam_size = kwargs.get("beam_size", 1)
        ctc_weight = kwargs.get("ctc_weight", 0.3)
        # Initialize Speech2Text model
        self.model = Speech2Text(
            s2t_train_config=self.s2t_train_config,
            s2t_model_file=self.s2t_model_file,
            bpemodel=self.bpemodel,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
            device=device,
        )
        self.text_prev = kwargs.get("text_prev", "<na>")
        self.lang_sym = kwargs.get("lang_sym", "<unk>")
        self.task_sym = kwargs.get("task_sym", "<pr>")
        self.return_ctc = kwargs.get("return_ctc", False)

    def infer(self, input_batch):
        with torch.no_grad():
            if "wav" in input_batch:
                speech = input_batch["wav"].squeeze(0).numpy()
            elif "wavpath" in input_batch:
                speech, _ = sf.read(input_batch["wavpath"])
            else:
                raise ValueError("input_batch must contain either 'wav' or 'wavpath'")

            # Override default parameters if provided in input_batch
            text_prev = input_batch.get("text_prev", self.text_prev)
            lang_sym = input_batch.get("lang_sym", self.lang_sym)
            task_sym = input_batch.get("task_sym", self.task_sym)

            # Run inference
            results = self.model(
                speech, text_prev=text_prev, lang_sym=lang_sym, task_sym=task_sym
            )

            # Return CTC states or transcription based on configuration
            if self.return_ctc:
                return results[-1].states["ctc"]
            else:
                transcription = results[0][0]
                print(transcription)
                return transcription


if __name__ == "__main__":
    wavpath = "/work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_buckeye/recording/s10_4829-0.flac"
    s2t_train_config = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/config.yaml"
    s2t_model_file = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/valid.acc.ave_5best.till40epoch.pth"
    bpemodel = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/data/token_list/bpe_unigram40000/bpe.model"
    beam_size = 1
    ctc_weight = 0.3
    text_prev = "<na>"
    lang_sym = "<unk>"
    task_sym = "<pr>"
    return_ctc = False
    inf = PowsmInference(
        device="cpu" if not torch.cuda.is_available() else "cuda",
        wavpath=wavpath,
        s2t_train_config=s2t_train_config,
        s2t_model_file=s2t_model_file,
        bpemodel=bpemodel,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        text_prev=text_prev,
        lang_sym=lang_sym,
        task_sym=task_sym,
        return_ctc=return_ctc,
    )
    result = inf.infer({"wavpath": wavpath})
    print(result)
