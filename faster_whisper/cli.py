import argparse
import json
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .transcribe import WhisperModel
from .tokenizer import _LANGUAGE_CODES
from .utils import download_model, format_timestamp

# Create a LANGUAGES dictionary for compatibility
LANGUAGES = {code: code for code in _LANGUAGE_CODES}


def str2bool(v):
    """Helper function for boolean command line arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def optional_int(v):
    """Helper function for optional integer arguments."""
    return None if v == "None" else int(v)


def optional_float(v):
    """Helper function for optional float arguments."""
    return None if v == "None" else float(v)


# Simple writer implementation for faster-whisper CLI
class SimpleWriter:
    def __init__(self, output_dir, output_format):
        self.output_dir = output_dir
        self.output_format = output_format
    
    def __call__(self, result, audio_path, **kwargs):
        import json
        
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        
        # Write text file
        if self.output_format in ["txt", "all"]:
            output_path = os.path.join(self.output_dir, audio_basename + ".txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
        
        # Write JSON file
        if self.output_format in ["json", "all"]:
            output_path = os.path.join(self.output_dir, audio_basename + ".json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Write SRT file
        if self.output_format in ["srt", "all"]:
            output_path = os.path.join(self.output_dir, audio_basename + ".srt")
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"], 1):
                    f.write(f"{i}\n")
                    f.write(f"{format_timestamp(segment['start'], True, ',')} --> {format_timestamp(segment['end'], True, ',')}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
        
        # Write VTT file
        if self.output_format in ["vtt", "all"]:
            output_path = os.path.join(self.output_dir, audio_basename + ".vtt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
        
        # Write TSV file
        if self.output_format in ["tsv", "all"]:
            output_path = os.path.join(self.output_dir, audio_basename + ".tsv")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("start\tend\ttext\n")
                for segment in result["segments"]:
                    f.write(f"{int(segment['start']*1000)}\t{int(segment['end']*1000)}\t{segment['text'].strip()}\n")


def get_writer(output_format, output_dir):
    """Simple writer factory for faster-whisper CLI."""
    return SimpleWriter(output_dir, output_format)


def cli():
    # List of available models
    available_models = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", 
        "distil-small.en", "medium", "medium.en", "distil-medium.en",
        "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", 
        "distil-large-v3", "large-v3-turbo", "turbo"
    ]

    def valid_model_name(name):
        if name in available_models or os.path.exists(name):
            return name
        raise ValueError(
            f"model should be one of {available_models} or path to a model checkpoint"
        )

    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="turbo", type=valid_model_name, help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    # Check if torch is available for default device selection
    try:
        import torch
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        default_device = "cpu"
    
    parser.add_argument("--device", default=default_device, help="device to use for PyTorch inference")
    parser.add_argument("--device_index", type=int, default=0, help="device index to use for GPU inference")
    parser.add_argument("--compute_type", type=str, default="default", choices=["default", "auto", "int8", "int8_float16", "int8_float32", "int8_bfloat16", "int16", "float16", "float32", "bfloat16"], help="compute type for ctranslate2 inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["txt", "vtt", "srt", "tsv", "json", "all"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in list(_LANGUAGE_CODES)]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="penalty applied to the score of previously generated tokens (set > 1 to penalize)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="prevent repetitions of ngrams with this size (set 0 to disable)")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--prefix", type=str, default=None, help="optional text to provide as a prefix for the first window.")

    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--prompt_reset_on_temperature", type=float, default=0.5, help="reset prompt if temperature is above this value")
    parser.add_argument("--without_timestamps", type=str2bool, default=False, help="only sample text tokens")
    parser.add_argument("--max_initial_timestamp", type=float, default=1.0, help="the initial timestamp cannot be later than this")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default='"\'"\u00bf([{-', help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default='"\'.。,，!！?？:：")]}、', help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--max_words_per_line", type=optional_int, default=None, help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--clip_timestamps", type=str, default="0", help="comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process, where the last end timestamp defaults to the end of the file")
    parser.add_argument("--hallucination_silence_threshold", type=optional_float, help="(requires --word_timestamps True) skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected")
    
    # faster-whisper specific arguments
    parser.add_argument("--vad_filter", type=str2bool, default=False, help="enable voice activity detection to filter out parts of the audio without speech")
    parser.add_argument("--vad_onset", type=float, default=0.5, help="VAD onset threshold")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="VAD offset threshold")
    parser.add_argument("--vad_min_speech_duration_ms", type=int, default=250, help="minimum speech duration in milliseconds")
    parser.add_argument("--vad_max_speech_duration_s", type=float, default=None, help="maximum speech duration in seconds")
    parser.add_argument("--vad_min_silence_duration_ms", type=int, default=2000, help="minimum silence duration in milliseconds")
    # fmt: on

    args = parser.parse_args().__dict__
    model_size_or_path: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    os.makedirs(output_dir, exist_ok=True)

    if model_size_or_path.endswith(".en") and args["language"] not in {"en", "English", None}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_size_or_path} is an English-only model but received '{args['language']}'; using English instead."
            )
        args["language"] = "en"

    # Convert language name to language code if needed
    if args["language"] is not None and args["language"].title() in _LANGUAGE_CODES:
        args["language"] = _LANGUAGE_CODES[args["language"].title()]

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.pop("threads")) > 0:
        try:
            import torch
            torch.set_num_threads(threads)
        except ImportError:
            # torch not available, ignore thread setting
            pass

    # Build VAD parameters if VAD is enabled
    vad_parameters = None
    if args["vad_filter"]:
        vad_parameters = {
            "threshold": args.pop("vad_onset"),
            "min_speech_duration_ms": args.pop("vad_min_speech_duration_ms"),
            "max_speech_duration_s": args.pop("vad_max_speech_duration_s"),
            "min_silence_duration_ms": args.pop("vad_min_silence_duration_ms"),
        }
    else:
        # Remove VAD parameters if VAD is not enabled
        for key in ["vad_onset", "vad_offset", "vad_min_speech_duration_ms", 
                    "vad_max_speech_duration_s", "vad_min_silence_duration_ms"]:
            args.pop(key, None)

    # Convert suppress_tokens string to list of integers
    suppress_tokens = args.pop("suppress_tokens")
    if suppress_tokens == "-1":
        suppress_tokens = [-1]
    elif suppress_tokens:
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
    else:
        suppress_tokens = None

    # Load the model
    model = WhisperModel(
        model_size_or_path,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        download_root=model_dir,
    )

    writer = get_writer(output_format, output_dir)
    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} requires --word_timestamps True")
    
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    if args["max_words_per_line"] and args["max_line_width"]:
        warnings.warn("--max_words_per_line has no effect with --max_line_width")
    
    writer_args = {arg: args.pop(arg) for arg in word_options}
    
    # Store args before processing
    audio_files = args.pop("audio")
    verbose = args.pop("verbose")
    
    # Process audio files
    for audio_path in audio_files:
        try:
            segments, info = model.transcribe(
                audio_path,
                language=args.get("language"),
                task=args.get("task"),
                beam_size=args.get("beam_size"),
                best_of=args.get("best_of"),
                patience=args.get("patience"),
                length_penalty=args.get("length_penalty"),
                repetition_penalty=args.get("repetition_penalty"),
                no_repeat_ngram_size=args.get("no_repeat_ngram_size"),
                temperature=temperature,
                compression_ratio_threshold=args.get("compression_ratio_threshold"),
                log_prob_threshold=args.get("logprob_threshold"),
                no_speech_threshold=args.get("no_speech_threshold"),
                condition_on_previous_text=args.get("condition_on_previous_text"),
                prompt_reset_on_temperature=args.get("prompt_reset_on_temperature"),
                initial_prompt=args.get("initial_prompt"),
                prefix=args.get("prefix"),
                suppress_blank=True,
                suppress_tokens=suppress_tokens,
                without_timestamps=args.get("without_timestamps"),
                max_initial_timestamp=args.get("max_initial_timestamp"),
                word_timestamps=args.get("word_timestamps"),
                prepend_punctuations=args.get("prepend_punctuations"),
                append_punctuations=args.get("append_punctuations"),
                vad_filter=args.get("vad_filter"),
                vad_parameters=vad_parameters,
                clip_timestamps=args.get("clip_timestamps"),
                hallucination_silence_threshold=args.get("hallucination_silence_threshold"),
                log_progress=verbose,
            )
            
            # Collect all segments into a list
            segments_list = list(segments)
            
            # Print transcription if verbose
            if verbose:
                print(f"\nDetected language: {info.language}")
                print(f"Transcription:")
                for segment in segments_list:
                    print(f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text.strip()}")
            
            # Convert segments to dict format for compatibility with writer
            result = {
                "text": " ".join([segment.text for segment in segments_list]).strip(),
                "segments": [
                    {
                        "id": segment.id,
                        "seek": segment.seek,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "tokens": segment.tokens,
                        "temperature": segment.temperature,
                        "avg_logprob": segment.avg_logprob,
                        "compression_ratio": segment.compression_ratio,
                        "no_speech_prob": segment.no_speech_prob,
                        "words": [
                            {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word,
                                "probability": word.probability,
                            }
                            for word in (segment.words or [])
                        ] if segment.words else None,
                    }
                    for segment in segments_list
                ],
                "language": info.language,
            }
            
            # Print summary
            if verbose:
                print(f"\nProcessed {len(segments_list)} segments")
                print(f"Saved transcription to {output_dir}")
            
            # Save the transcription
            writer(result, audio_path, **writer_args)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {type(e).__name__}: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    cli() 