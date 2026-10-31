# InteractiveOmni-8B Cog Model

This project packages the [`sensefvg/InteractiveOmni-8B`](https://huggingface.co/sensefvg/InteractiveOmni-8B) multi-modal conversational model for Replicate using [Cog](https://github.com/replicate/cog). InteractiveOmni-8B can accept combinations of text, images, audio, and video inputs and respond with rich text — optionally accompanied by synthesized speech.

## Features

- ✅ Supports text-only, image-grounded, audio-grounded, and video-grounded prompts.
- 🎙️ Optional speech generation (24 kHz WAV) using the model's built-in TTS head.
- 🧠 Automatic insertion of the recommended system prompt when audio generation is requested.
- 📦 Ships with locally cached model weights (~19 GB) for offline builds.

## Project Layout

```
omni/
├── cog.yaml                 # Cog build configuration
├── predict.py               # Predictor implementation
├── requirements.txt         # Python dependencies
└── checkpoints/
    └── InteractiveOmni-8B/  # Downloaded model weights and tokenizer
```

## Inputs

| Name | Type | Description |
| --- | --- | --- |
| `prompt` | `string` | Optional user text prompt. Leave blank when providing only media. |
| `images` | `List[Path]` | Optional list of images (`.png`, `.jpg`, `.webp`). |
| `audio` | `Path` | Optional audio clip (`.wav`, `.mp3`, `.flac`). Automatically resampled to 24 kHz mono. |
| `video` | `Path` | Optional video (`.mp4`, `.mov`, `.webm`). |
| `enable_audio_output` | `bool` | When true, returns synthesized speech alongside text. |
| `system_prompt` | `string` | Optional system message. If omitted while `enable_audio_output` is true, a recommended prompt is injected. |
| `max_new_tokens` | `int` | Maximum generated tokens (default 512). |
| `temperature` | `float` | Sampling temperature (0 for greedy). |
| `top_p` | `float` | Top-p nucleus sampling value. |
| `max_tiles` | `int` | Max temporal tiles extracted from video (default 12). |
| `frames_per_tile` | `int` | Frames sampled per tile (default 8). |
| `seed` | `int` | RNG seed (`-1` to disable). |

## Outputs

- `text`: string containing the generated response.
- `audio`: optional path to a 24 kHz WAV file when `enable_audio_output` is true and the model produces speech.

## Usage

### Local testing

```bash
cd omni
# Ensure Cog is installed: pip install cog
cog predict -i prompt="Hello! Who are you?"
```

### With media

```bash
cog predict \
  -i prompt="Describe the scene." \
  -i images=@examples/beach.jpg
```

```bash
cog predict \
  -i audio=@hey.mp3\
  -i enable_audio_output=true
```

```bash
cog predict \
  -i video=@monkey.mp4 \
  -i prompt="Summarize the main action." \
  -i max_tiles=12 \
  -i frames_per_tile=8
```

## Build for Replicate

```bash
cd omni
cog build
```

The build uses CUDA 12.1, Python 3.10, and installs dependencies from `requirements.txt`. Model weights are included via the `weights` stanza in `cog.yaml`, so the resulting image contains everything required for inference.

## Model Notes

- **License:** MIT (as published on the Hugging Face model card).
- **Recommended system prompt for speech output:**
  ```
  You are a highly advanced multimodal conversational AI designed for human-like interaction. You can perceive auditory, visual, speech, and textual inputs, and generate text and speech.
  ```
  This is automatically inserted when `enable_audio_output` is true and no custom system prompt is provided.
- **Hardware:** Designed for GPU inference. The predictor automatically selects BF16, FP16, or FP32 depending on device support.

## Acknowledgements

InteractiveOmni is developed by the [SenseTime Future Vision Group](https://github.com/SenseTime-FVG). Please cite their work if you use this project in research.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. [sensenova/InteractiveOmni-8B](https://huggingface.co/sensenova/InteractiveOmni-8B) is released under the [MIT](https://github.com/SenseTime-FVG/InteractiveOmni/blob/main/LICENSE) license.
