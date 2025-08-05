# File: engine.py
# Core TTS model loading and speech generation logic for KittenTTS ONNX.

import torch
import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple
from pathlib import Path
from huggingface_hub import hf_hub_download
import phonemizer

# ... other imports
import espeakng_loader

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
onnx_session: Optional[ort.InferenceSession] = None
voices_data: Optional[dict] = None
phonemizer_backend: Optional[phonemizer.backend.EspeakBackend] = None
text_cleaner: Optional["TextCleaner"] = None
MODEL_LOADED: bool = False

# KittenTTS available voices
KITTEN_TTS_VOICES = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]


class TextCleaner:
    """Text cleaner for KittenTTS - converts text to token indices."""

    def __init__(self):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

        self.word_index_dictionary = {}
        for i in range(len(symbols)):
            self.word_index_dictionary[symbols[i]] = i

    def __call__(self, text: str):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


def basic_english_tokenize(text: str):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


def load_model() -> bool:
    """
    Loads the KittenTTS model from Hugging Face Hub and initializes ONNX session.
    Updates global variables for model components.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global onnx_session, voices_data, phonemizer_backend, text_cleaner, MODEL_LOADED

    if MODEL_LOADED:
        logger.info("KittenTTS model is already loaded.")
        return True

    try:
        # Get model repository and cache path from config
        model_repo_id = config_manager.get_string(
            "model.repo_id", "KittenML/kitten-tts-nano-0.1"
        )
        model_cache_path = config_manager.get_path(
            "paths.model_cache", "./model_cache", ensure_absolute=True
        )

        logger.info(f"Loading KittenTTS model from: {model_repo_id}")
        logger.info(f"Using cache directory: {model_cache_path}")

        # Ensure cache directory exists
        model_cache_path.mkdir(parents=True, exist_ok=True)

        # Download config.json first
        config_path = hf_hub_download(
            repo_id=model_repo_id,
            filename="config.json",
            cache_dir=str(model_cache_path),
        )

        # Load config to get model filenames
        import json

        with open(config_path, "r") as f:
            model_config = json.load(f)

        if model_config.get("type") != "ONNX1":
            raise ValueError("Unsupported model type. Expected ONNX1.")

        # Download model and voices files
        model_path = hf_hub_download(
            repo_id=model_repo_id,
            filename=model_config["model_file"],
            cache_dir=str(model_cache_path),
        )

        voices_path = hf_hub_download(
            repo_id=model_repo_id,
            filename=model_config["voices"],
            cache_dir=str(model_cache_path),
        )

        # Load voices data
        voices_data = np.load(voices_path)
        logger.info(f"Loaded voices data with keys: {list(voices_data.keys())}")

        # Determine device and providers and configure for optimal performance
        device_setting = config_manager.get_string("tts_engine.device", "auto").lower()
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")

        sess_options = ort.SessionOptions()
        providers = []
        provider_options = []

        # Priority: Check for requested GPU first.
        if device_setting in ["cuda", "gpu"]:
            if "CUDAExecutionProvider" in available_providers:
                logger.info(
                    "Configuration requests GPU, and CUDAExecutionProvider is available."
                )
                logger.info(
                    "Configuring CUDAExecutionProvider with pinned memory for optimal performance."
                )
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                provider_options = [
                    {
                        "device_id": "0",
                        "gpu_mem_type": "pinned",
                    },
                    {},
                ]
            else:
                logger.warning(
                    "Configuration requests GPU, but CUDAExecutionProvider is NOT available. "
                    "Ensure you have a compatible NVIDIA driver, CUDA Toolkit, and cuDNN installed. "
                    "Falling back to CPU."
                )
                providers = ["CPUExecutionProvider"]

        # If providers list is still empty, it means GPU was not requested. Default to CPU.
        if not providers:
            logger.info("Using ONNX Runtime with CPUExecutionProvider.")
            providers = ["CPUExecutionProvider"]

        # Initialize the ONNX Inference Session with the chosen providers and options
        logger.info(
            f"Initializing ONNX InferenceSession from {model_path} with providers: {providers}"
        )
        onnx_session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=providers,
            provider_options=provider_options,
        )

        # Auto-configure eSpeak for Windows
        if os.name == "nt":  # Windows
            import platform

            # Common eSpeak installation paths on Windows
            possible_paths = [
                Path(r"C:\Program Files\eSpeak NG"),
                Path(r"C:\Program Files (x86)\eSpeak NG"),
                Path(r"C:\eSpeak NG"),
                Path(os.environ.get("ProgramFiles", "")) / "eSpeak NG",
                Path(os.environ.get("ProgramFiles(x86)", "")) / "eSpeak NG",
            ]

            espeak_found = False
            for espeak_path in possible_paths:
                if espeak_path.exists():
                    dll_path = espeak_path / "libespeak-ng.dll"
                    data_path = espeak_path / "espeak-ng-data"

                    if dll_path.exists():
                        # Set environment variables for phonemizer
                        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(dll_path)
                        os.environ["PHONEMIZER_ESPEAK_PATH"] = str(
                            espeak_path / "espeak-ng.exe"
                        )

                        # Configure phonemizer's EspeakWrapper
                        from phonemizer.backend.espeak.wrapper import (
                            EspeakWrapper as PhonemizeEspeakWrapper,
                        )

                        PhonemizeEspeakWrapper.set_library(str(dll_path))

                        logger.info(f"Auto-configured eSpeak from: {espeak_path}")
                        espeak_found = True
                        break

            if not espeak_found:
                logger.warning(
                    "eSpeak NG not found in common locations on Windows. "
                    "Please install from: https://github.com/espeak-ng/espeak-ng/releases "
                    "Choose the .msi installer for your system (usually 64-bit)."
                )
                # Try to proceed anyway - espeakng_loader might find it

        try:
            # Import only when needed to avoid issues if misaki is not fully installed
            import misaki.espeak

            espeak_data_path = espeakng_loader.get_data_path()
            # Check if the class and the specific attribute exist before setting
            if hasattr(misaki.espeak, "EspeakWrapper") and hasattr(
                misaki.espeak.EspeakWrapper, "data_path"
            ):
                misaki.espeak.EspeakWrapper.data_path = espeak_data_path
                logger.info(
                    f"eSpeak data path configured for misaki: {espeak_data_path}"
                )
            else:
                logger.warning(
                    "misaki.espeak.EspeakWrapper.data_path attribute not found. Skipping configuration."
                )
        except Exception as e:
            logger.warning(f"Could not auto-configure misaki eSpeak data path: {e}")

        # Initialize phonemizer with better error handling
        try:
            # Suppress phonemizer warnings during initialization
            import logging as log_module

            phonemizer_logger = log_module.getLogger("phonemizer")
            original_level = phonemizer_logger.level
            phonemizer_logger.setLevel(log_module.ERROR)

            phonemizer_backend = phonemizer.backend.EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True
            )

            phonemizer_logger.setLevel(original_level)
            logger.info("Phonemizer backend initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize phonemizer: {e}")
            logger.error(
                "Please ensure eSpeak NG is installed:\n"
                "  Windows: Download from https://github.com/espeak-ng/espeak-ng/releases\n"
                "  Linux: Run 'sudo apt install espeak-ng'"
            )
            raise

        # Initialize text cleaner
        text_cleaner = TextCleaner()

        MODEL_LOADED = True
        logger.info("KittenTTS model loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"Error loading KittenTTS model: {e}", exc_info=True)
        onnx_session = None
        voices_data = None
        phonemizer_backend = None
        text_cleaner = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str, voice: str, speed: float = 1.0
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Synthesizes audio from text using the loaded KittenTTS model.

    Args:
        text: The text to synthesize.
        voice: Voice identifier (e.g., 'expr-voice-5-m').
        speed: Speech speed factor (1.0 is normal speed).

    Returns:
        A tuple containing the audio waveform (numpy array) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global onnx_session, voices_data, phonemizer_backend, text_cleaner

    if not MODEL_LOADED or onnx_session is None:
        logger.error("KittenTTS model is not loaded. Cannot synthesize audio.")
        return None, None

    if voice not in KITTEN_TTS_VOICES:
        logger.error(
            f"Voice '{voice}' not available. Available voices: {KITTEN_TTS_VOICES}"
        )
        return None, None

    try:
        logger.debug(f"Synthesizing with voice='{voice}', speed={speed}")
        logger.debug(f"Input text (first 100 chars): '{text[:100]}...'")

        # Phonemize the input text
        # Suppress the word count mismatch warning by temporarily adjusting log level
        import logging as log_module

        phonemizer_logger = log_module.getLogger("phonemizer")
        original_level = phonemizer_logger.level
        phonemizer_logger.setLevel(log_module.ERROR)

        phonemes_list = phonemizer_backend.phonemize([text])

        # Restore original log level
        phonemizer_logger.setLevel(original_level)

        # Process phonemes to get token IDs
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = " ".join(phonemes)
        tokens = text_cleaner(phonemes)

        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(0)

        # Determine the execution device from the session to decide where to place tensors
        provider = onnx_session.get_providers()[0]

        if provider == "CUDAExecutionProvider":
            # --- I/O Binding Path for GPU ---
            # Create torch tensors directly on the GPU
            device = "cuda"
            input_ids = torch.tensor([tokens], dtype=torch.int64, device=device)
            ref_s = torch.tensor(voices_data[voice], device=device)
            speed_array = torch.tensor([speed], dtype=torch.float32, device=device)

            # Create OrtValues from torch tensors without copying data
            input_ids_ort = ort.OrtValue.ortvalue_from_pytorch(input_ids)
            ref_s_ort = ort.OrtValue.ortvalue_from_pytorch(ref_s)
            speed_array_ort = ort.OrtValue.ortvalue_from_pytorch(speed_array)

            # Set up I/O binding
            io_binding = onnx_session.io_binding()

            # Bind inputs
            io_binding.bind_ortvalue_input("input_ids", input_ids_ort)
            io_binding.bind_ortvalue_input("style", ref_s_ort)
            io_binding.bind_ortvalue_input("speed", speed_array_ort)

            # Bind output to the GPU
            io_binding.bind_output("output", "cuda")

            # Run inference with binding
            output_ortvalue = onnx_session.run_with_iobinding(io_binding)[0]

            # The output is on the GPU. Copy it back to the CPU to be used by the rest of the app.
            # This is the ONLY data copy in the GPU path.
            audio = output_ortvalue.numpy()

        else:
            # --- Standard Path for CPU ---
            input_ids = np.array([tokens], dtype=np.int64)
            ref_s = voices_data[voice]
            speed_array = np.array([speed], dtype=np.float32)

            onnx_inputs = {
                "input_ids": input_ids,
                "style": ref_s,
                "speed": speed_array,
            }
            # Run standard inference
            outputs = onnx_session.run(None, onnx_inputs)
            audio = outputs[0]

        # KittenTTS uses 24kHz sample rate
        sample_rate = 24000

        logger.info(
            f"Successfully generated {len(audio)} audio samples at {sample_rate}Hz"
        )
        return audio, sample_rate

    except Exception as e:
        logger.error(f"Error during KittenTTS synthesis: {e}", exc_info=True)
        return None, None


# --- End File: engine.py ---
