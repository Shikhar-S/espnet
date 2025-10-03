import torch
import soundfile as sf
import tempfile


class AllophantInference:
    def __init__(self, device="cpu", **kwargs):
        # allophant imports use old versions of various libraries and are troublesome in general
        from allophant.estimator import Estimator
        from allophant.config import PhonemeLayerType

        self.device = device
        self.model, self.training_attribute_indexer = Estimator.restore(
            "kgnlp/allophant", device=self.device
        )
        self.lang2cache = {}
        self.uses_allophones = (
            self.model.config.nn.projection.phoneme_layer == PhonemeLayerType.ALLOPHONES
        )
        self.MISSING_LANGUAGES = [
            "ara",
            "aze",
            "bak",
            "bej",
            "bel",
            "boa",
            "bos",
            "cym",
            "ful",
            "jje",
            "kaz",
            "kmr",
            "mal",
            "mar",
            "mon",
            "msa",
            "nya",
            "ori",
            "orm",
            "pbv",
            "srp",
            "sva",
            "tat",
            "tgk",
            "tio",
            "uig",
            "urd",
            "uum",
            "uzb",
            "yno",
        ]

    def get_decoder_for_language(self, language):
        if language in self.lang2cache:
            return self.lang2cache[language]
        from allophant import predictions

        inventory = self.training_attribute_indexer.phoneme_inventory(
            languages=language
        )
        attribute_indexer = self.training_attribute_indexer
        if not inventory:
            from allophant.phonetic_features import PhoneticAttributeIndexer
            from allophant.config import FeatureSet

            print(
                "This seems like an out of training distribution language. We will create a zero-shot inventory."
            )
            attribute_indexer = PhoneticAttributeIndexer(
                feature_set=FeatureSet.PHOIBLE,
                attribute_subset=self.training_attribute_indexer.feature_names,
                phoneme_subset=None,
                language_inventories=None,
                allophones_from_allophoible=self.uses_allophones,
            )
            inventory = attribute_indexer.phoneme_inventory(languages=language)
            if len(inventory) == 0:
                # clear cache
                self.lang2cache = {}
                self.lang2cache[language] = (None, None, None)
                return None, None, None
        feature_matrix = attribute_indexer.composition_feature_matrix(inventory).to(
            self.device
        )
        inventory_indexer = attribute_indexer.attributes.subset(inventory)
        decoder = predictions.feature_decoders(
            inventory_indexer, feature_names=["phoneme"]
        )["phoneme"]
        # clear cache
        self.lang2cache = {}
        self.lang2cache[language] = (feature_matrix, decoder, inventory_indexer)
        return feature_matrix, decoder, inventory_indexer

    def infer(self, input_batch):
        if input_batch.get("language", None) in self.MISSING_LANGUAGES:
            print(
                f"Warning: Language {input_batch.get('language', None)} is known to be missing. Skipping key {input_batch['key']}."
            )
            return ""
        from allophant.dataset_processing import Batch

        with torch.no_grad():
            audio_input = input_batch["wav"]
            assert (
                audio_input.shape[0] == 1
            ), "Batch size > 1 not supported for inference!"
            feature_matrix, decoder, inventory_indexer = self.get_decoder_for_language(
                input_batch.get("language", None)
            )
            if feature_matrix is None:
                print(
                    f"Warning: No inventory found for language {input_batch.get('language', None)}. Skipping key {input_batch['key']}."
                )
                return ""
            batch = Batch(
                audio_input, torch.tensor([audio_input.shape[1]]), torch.zeros(1)
            ).to(self.device)
            # .outputs['phoneme'] contains the logits over inventory
            model_outputs = self.model.predict(batch, feature_matrix)
            decoded = decoder(
                model_outputs.outputs["phoneme"].transpose(1, 0), model_outputs.lengths
            )
            for [hypothesis] in decoded:
                recognized = inventory_indexer.feature_values(
                    "phoneme", hypothesis.tokens - 1
                )
                recognized = "".join(recognized)
        return recognized


class AllosaurusInference:
    def __init__(self, device="cpu", **kwargs):
        from allosaurus.app import read_recognizer

        self.device = device
        self.model = read_recognizer()
        if device == "cuda" and torch.cuda.is_available():
            self.model.config.device_id = 0
            self.model.am.to("cuda")
        else:
            self.model.config.device_id = -1  # CPU

    def infer(self, input_batch):
        with torch.no_grad():
            audiopath = input_batch["wavpath"]
            if not audiopath.endswith(".wav"):
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=True
                ) as temp_wav:
                    sf.write(
                        temp_wav.name, input_batch["wav"].squeeze(0).numpy(), 16000
                    )
                    transcription = self.model.recognize(temp_wav.name)
            else:
                transcription = self.model.recognize(audiopath)
        recognized = "".join(transcription)
        return recognized


class Wav2Vec2PhonemeInference:
    def __init__(self, device="cpu", **kwargs):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        self.device = device
        self.model_path = kwargs.get("model_path", "")
        assert (
            self.model_path != ""
        ), "model_path must be specified for Wav2Vec2PhonemeInference"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        # Load model with appropriate dtype based on device
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_path, torch_dtype=torch.float32
        )
        self.dtype = torch.float32
        self.model.eval().to(device)

    def infer(self, input_batch):
        audio = input_batch["wav"].squeeze(0).numpy()
        inputs = self.processor(
            audio,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            input_values = inputs.input_values.to(self.device)
            input_values = input_values.to(self.dtype)
            if hasattr(inputs, "attention_mask"):
                attention_mask = inputs.attention_mask.to(self.device)
                logits = self.model(input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        recognized = "".join(transcription)
        return recognized


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
            lang_sym = input_batch.get("language", self.lang_sym)
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
                transcription = transcription.split(">")[-1].replace("/", "")
                return transcription


def get_inference_model(model_name, device="cpu", **kwargs):
    if model_name.startswith("powsm"):
        return PowsmInference(device=device, **kwargs)
    elif model_name == "allophant":
        return AllophantInference(device=device, **kwargs)
    elif model_name == "allosaurus":
        return AllosaurusInference(device=device, **kwargs)
    elif model_name in [
        "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
    ]:
        return Wav2Vec2PhonemeInference(model_path=model_name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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
