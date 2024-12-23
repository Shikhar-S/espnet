from espnet2.sds.utils.utils import int2float


def handle_espnet_TTS_intelligibility(TTS_audio_output, LLM_Output):
    try:
        from versa import (
            espnet_levenshtein_metric,
            espnet_wer_setup,
            owsm_levenshtein_metric,
            owsm_wer_setup,
            whisper_levenshtein_metric,
            whisper_wer_setup,
        )
    except Exception as e:
        print("Error: Versa is not properly installed.")
        raise e
    score_modules_espnet = {
        "module": espnet_levenshtein_metric,
        "args": espnet_wer_setup(
            model_tag="default",
            beam_size=1,
            text_cleaner="whisper_en",
            use_gpu=True,
        ),
    }
    dict1 = score_modules_espnet["module"](
        score_modules_espnet["args"],
        int2float(TTS_audio_output[1]),
        LLM_Output,
        TTS_audio_output[0],
    )
    espnet_wer = (
        dict1["espnet_wer_delete"]
        + dict1["espnet_wer_insert"]
        + dict1["espnet_wer_replace"]
    ) / (
        dict1["espnet_wer_delete"]
        + dict1["espnet_wer_replace"]
        + dict1["espnet_wer_equal"]
    )
    espnet_cer = (
        dict1["espnet_cer_delete"]
        + dict1["espnet_cer_insert"]
        + dict1["espnet_cer_replace"]
    ) / (
        dict1["espnet_cer_delete"]
        + dict1["espnet_cer_replace"]
        + dict1["espnet_cer_equal"]
    )
    score_modules_owsm = {
        "module": owsm_levenshtein_metric,
        "args": owsm_wer_setup(
            model_tag="default",
            beam_size=1,
            text_cleaner="whisper_en",
            use_gpu=True,
        ),
    }
    dict1 = score_modules_owsm["module"](
        score_modules_owsm["args"],
        int2float(TTS_audio_output[1]),
        LLM_Output,
        TTS_audio_output[0],
    )
    owsm_wer = (
        dict1["owsm_wer_delete"] + dict1["owsm_wer_insert"] + dict1["owsm_wer_replace"]
    ) / (dict1["owsm_wer_delete"] + dict1["owsm_wer_replace"] + dict1["owsm_wer_equal"])
    owsm_cer = (
        dict1["owsm_cer_delete"] + dict1["owsm_cer_insert"] + dict1["owsm_cer_replace"]
    ) / (dict1["owsm_cer_delete"] + dict1["owsm_cer_replace"] + dict1["owsm_cer_equal"])
    score_modules_whisper = {
        "module": whisper_levenshtein_metric,
        "args": whisper_wer_setup(
            model_tag="default",
            beam_size=1,
            text_cleaner="whisper_en",
            use_gpu=True,
        ),
    }
    dict1 = score_modules_whisper["module"](
        score_modules_whisper["args"],
        int2float(TTS_audio_output[1]),
        LLM_Output,
        TTS_audio_output[0],
    )
    whisper_wer = (
        dict1["whisper_wer_delete"]
        + dict1["whisper_wer_insert"]
        + dict1["whisper_wer_replace"]
    ) / (
        dict1["whisper_wer_delete"]
        + dict1["whisper_wer_replace"]
        + dict1["whisper_wer_equal"]
    )
    whisper_cer = (
        dict1["whisper_cer_delete"]
        + dict1["whisper_cer_insert"]
        + dict1["whisper_cer_replace"]
    ) / (
        dict1["whisper_cer_delete"]
        + dict1["whisper_cer_replace"]
        + dict1["whisper_cer_equal"]
    )
    return (
        f"ESPnet WER: {espnet_wer*100:.2f}\n"
        "ESPnet CER: {espnet_cer*100:.2f}\n"
        "OWSM WER: {owsm_wer*100:.2f}\n"
        "OWSM CER: {owsm_cer*100:.2f}\n"
        "Whisper WER: {whisper_wer*100:.2f}\n"
        "Whisper CER: {whisper_cer*100:.2f}"
    )
