import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import time
import subprocess
import torch
import torchaudio
from cog import BaseModel, BasePredictor, Input, Path as CogPath
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_REPO_ID = "sensefvg/InteractiveOmni-8B"
MODEL_LOCAL_DIR = Path("checkpoints")
TARGET_SAMPLE_RATE = 24000
DEFAULT_SYSTEM_PROMPT = (
    "You are a highly advanced multimodal conversational AI designed for human-like "
    "interaction. You can perceive auditory, visual, speech, and textual inputs, and "
    "generate text and speech."
)
MODEL_URL = "https://weights.replicate.delivery/default/sensefvg/InteractiveOmni-8B/model.tar"


class OmniOutput(BaseModel):
    text: str
    audio: Optional[CogPath] = None

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download model weights and load the model into memory."""
        if not os.path.exists(MODEL_LOCAL_DIR):
            download_weights(MODEL_URL, MODEL_LOCAL_DIR)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        config_path = MODEL_LOCAL_DIR / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
            llm_cfg = config_data.get("llm_config", {})
            if llm_cfg.get("attn_implementation") == "flash_attention_2":
                llm_cfg["attn_implementation"] = "sdpa"
                attn_cfg = llm_cfg.get("attn_config")
                if isinstance(attn_cfg, dict):
                    attn_cfg["attn_implementation"] = "sdpa"
                config_data["llm_config"] = llm_cfg
            if config_data.get("attn_implementation") == "flash_attention_2":
                config_data["attn_implementation"] = "sdpa"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_LOCAL_DIR),
            trust_remote_code=True,
            use_fast=True,
        )
        config = AutoConfig.from_pretrained(
            str(MODEL_LOCAL_DIR),
            trust_remote_code=True,
        )

        llm_config = getattr(config, "llm_config", None)
        if llm_config is not None:
            attn_impl = getattr(llm_config, "attn_implementation", None)
            if attn_impl == "flash_attention_2":
                setattr(llm_config, "attn_implementation", "sdpa")
                attn_cfg = getattr(llm_config, "attn_config", None)
                if isinstance(attn_cfg, dict):
                    attn_cfg["attn_implementation"] = "sdpa"

        if getattr(config, "attn_implementation", None) == "flash_attention_2":
            setattr(config, "attn_implementation", "sdpa")

        self.model = AutoModel.from_pretrained(
            str(MODEL_LOCAL_DIR),
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None,
            config=config,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="interactiveomni-"))

    def predict(
        self,
        prompt: str = Input(
            description="User text prompt. Leave blank when providing only media inputs.",
            default="",
        ),
        images: Optional[List[CogPath]] = Input(
            description="Optional list of images (PNG/JPG/WebP) to provide visual context.",
            default=None,
        ),
        audio: Optional[CogPath] = Input(
            description="Optional audio clip (WAV/MP3/FLAC). Resampled to 24 kHz automatically.",
            default=None,
        ),
        video: Optional[CogPath] = Input(
            description="Optional video clip (MP4/MOV/WebM) for video-grounded conversation.",
            default=None,
        ),
        enable_audio_output: bool = Input(
            description="Return generated speech along with text output.",
            default=False,
        ),
        system_prompt: str = Input(
            description=(
                "Optional system prompt. When audio output is enabled and this is left blank, "
                "a recommended prompt is injected automatically."
            ),
            default="",
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate.",
            ge=32,
            le=2048,
            default=512,
        ),
        temperature: float = Input(
            description="Sampling temperature. Set to 0 for greedy decoding.",
            ge=0.0,
            le=2.0,
            default=0.7,
        ),
        top_p: float = Input(
            description="Top-p nucleus sampling parameter.",
            ge=0.01,
            le=1.0,
            default=0.8,
        ),
        max_tiles: int = Input(
            description="Maximum number of temporal tiles to sample when a video is provided.",
            ge=1,
            le=48,
            default=12,
        ),
        frames_per_tile: int = Input(
            description="Number of frames sampled per tile when processing video.",
            ge=1,
            le=32,
            default=8,
        ),
        seed: int = Input(
            description="Random seed for reproducible sampling. Set to -1 to disable seeding.",
            default=0,
        ),
    ) -> OmniOutput:
        if not any([prompt.strip(), images, audio, video]):
            raise ValueError("Provide at least one of prompt, images, audio, or video inputs.")

        messages: List[Dict[str, object]] = []
        system_prompt = system_prompt.strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif enable_audio_output:
            messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

        user_content: List[Dict[str, str]] = []

        if images:
            for idx, image in enumerate(images):
                image_path = Path(str(image))
                if not image_path.exists():
                    raise FileNotFoundError(f"Image input #{idx + 1} not found at {image_path}")
                user_content.append({"type": "image", "image": str(image_path)})

        if audio:
            prepared_audio = self._prepare_audio(Path(str(audio)))
            user_content.append({"type": "audio", "audio": prepared_audio})

        if video:
            video_path = Path(str(video))
            if not video_path.exists():
                raise FileNotFoundError(f"Video input not found at {video_path}")
            user_content.append({"type": "video", "video": str(video_path)})

        if prompt.strip():
            user_content.append({"type": "text", "text": prompt.strip()})

        if not user_content:
            raise ValueError("At least one content input must be provided for the user message.")

        messages.append({"role": "user", "content": user_content})

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }

        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        chat_kwargs: Dict[str, object] = {}
        if video:
            chat_kwargs["max_patch_num"] = max_tiles
            chat_kwargs["frame"] = frames_per_tile

        response_text: Optional[str] = None
        response_audio: Optional[CogPath] = None
        if enable_audio_output:
            text, audio_tensor = self.model.chat(
                self.tokenizer,
                generation_config,
                messages,
                generate_audio=True,
                **chat_kwargs,
            )
            response_text = str(text)
            if audio_tensor is not None:
                audio_out_path = self._tmp_dir / "omni_response.wav"
                torchaudio.save(
                    str(audio_out_path),
                    audio_tensor.cpu(),
                    TARGET_SAMPLE_RATE,
                    format="wav",
                )
                response_audio = CogPath(str(audio_out_path))
        else:
            text = self.model.chat(self.tokenizer, generation_config, messages, **chat_kwargs)
            response_text = str(text)

        if response_text is None:
            response_text = ""

        return OmniOutput(text=response_text, audio=response_audio)

    def _prepare_audio(self, audio_path: Path) -> str:
        """Convert audio input to 24 kHz mono WAV expected by the model."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio input not found at {audio_path}")

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.ndim > 1 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        output_path = self._tmp_dir / f"audio_{audio_path.stem}_24k.wav"
        torchaudio.save(str(output_path), waveform, TARGET_SAMPLE_RATE, format="wav")
        return str(output_path)
