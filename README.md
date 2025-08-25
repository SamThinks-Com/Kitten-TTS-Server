# Kitten TTS Server — Ultra-Light TTS API, Web UI, GPU Ready

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/SamThinks-Com/Kitten-TTS-Server/releases)

![Kitten TTS Banner](https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=1600&q=80)

Kitten-TTS-Server hosts the Kitten TTS model behind a small, fast API. It includes a built-in Web UI, audiobook-grade text handling, and GPU support. Use it for prototypes, research, or self-hosted TTS services.

Badges
- Topics: ai, api-server, audio-generation, cuda, fastapi, huggingface, kitten, kitten-tts, kittentts, openai-api, python, pytorch, speech-synthesis, text-to-speech, tts, tts-api, web-ui
- Platform: Linux, macOS, Windows (WSL recommended for CUDA)
- Languages: Python, Bash

Table of contents
- Features
- Quick links
- Requirements
- Installation (direct release)
- Docker / Docker Compose
- Configuration
- Run the server
- Web UI
- API reference
- Large-text / audiobook mode
- GPU tips and CUDA setup
- Model management and Hugging Face
- OpenAI-style endpoint proxy
- Streaming audio and real-time
- Security and access
- Logging and metrics
- Benchmarks and perf tuning
- Examples (curl, Python)
- CLI client
- Systemd and service deploy
- Troubleshooting
- Contributing
- License
- Acknowledgements
- Assets and images

Features
- Lightweight server built on FastAPI and Uvicorn.
- Kitten TTS model integration (small footprint).
- Local or Hugging Face hosted model support.
- Audiobook-grade text chunking with pause control.
- GPU acceleration via PyTorch / CUDA.
- Web UI that runs in the browser to test voices and export audio.
- OpenAI-like API endpoints for drop-in integration.
- Streaming endpoints for progressive playback.
- Docker images for quick deployment.
- Modular code for custom models and backends.

Quick links
- Releases (download and run the installer): [![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/SamThinks-Com/Kitten-TTS-Server/releases)
- Repository: https://github.com/SamThinks-Com/Kitten-TTS-Server
- Hugging Face: model support via huggingface transformers or diffusers for TTS backends.

NOTE: The Releases link above points to packaged builds. Download the release archive or installer from that page and execute the included installer file to set up a ready build on your host.

Requirements
- Python 3.10 or 3.11
- pip
- Git
- For GPU: CUDA-compatible GPU, appropriate CUDA toolkit for your PyTorch build
- For Docker: Docker Engine 20+ and docker-compose 1.29+

Minimum recommended specs for decent performance
- CPU: 4 cores
- RAM: 8 GB
- Disk: 2 GB free
- GPU: NVIDIA card with 8+ GB VRAM for comfortable GPU runs

Installation (direct release)
1. Visit the Releases page and download the release file. The release may come as an archive (tar.gz, zip) or an installer script. Use the link below.
   - https://github.com/SamThinks-Com/Kitten-TTS-Server/releases
2. Extract the archive.
   - Example: tar xf Kitten-TTS-Server-vX.Y.Z.tar.gz
3. Run the installer or setup script inside the release package.
   - Example: cd Kitten-TTS-Server-vX.Y.Z && ./install.sh
4. If the release provides a binary executable, make it executable and run it.
   - Example: chmod +x kitten-tts-server && ./kitten-tts-server
5. If you prefer to install from source, follow the Source install steps below.

Source install (pip)
- Clone the repo:
  - git clone https://github.com/SamThinks-Com/Kitten-TTS-Server.git
  - cd Kitten-TTS-Server
- Create a venv:
  - python -m venv .venv
  - source .venv/bin/activate
- Install dependencies:
  - pip install -r requirements.txt
  - pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121  # choose correct index for your CUDA
- Install the package:
  - pip install -e .

Docker / Docker Compose
- Use Docker to isolate the runtime.
- Example Dockerfile is included in the repo. Build and run:
  - docker build -t kitten-tts-server:latest .
  - docker run --gpus all -p 8000:8000 --rm kitten-tts-server:latest
- Example docker-compose.yml:
  - version: "3.8"
    services:
      kitten:
        image: kitten-tts-server:latest
        ports:
          - "8000:8000"
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]
- For CPU-only runs, remove the --gpus flag and skip CUDA-specific python wheels.

Configuration
- The server reads configuration from environment variables or a config file.
- Common env variables:
  - KITTEN_MODEL_PATH: path to local model or Hugging Face repo ID.
  - KITTEN_DEVICE: "cpu" or "cuda"
  - HOST: default 0.0.0.0
  - PORT: default 8000
  - MAX_TEXT_CHUNK: max characters per chunk for audiobook mode (default 4000)
  - SAMPLE_RATE: output audio sample rate (default 22050)
  - VOICE: name of voice preset
  - OPENAI_KEY: optional key to gate OpenAI-style proxy
  - CORS_ORIGINS: comma separated allowed origins
- Use .env file to store env vars for local dev. Example .env:
  - HOST=0.0.0.0
  - PORT=8000
  - KITTEN_DEVICE=cuda
  - KITTEN_MODEL_PATH=facebook/kitten-tts-small
  - MAX_TEXT_CHUNK=3500

Run the server
- Development:
  - uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
- Production (example using uvicorn with workers):
  - uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
- When the server starts, it logs the model load and device assignment. The Web UI appears at http://localhost:8000/ui

Web UI
- The Web UI provides a simple console to test voices and export files.
- Features:
  - Text input with large-text paste.
  - Voice selection.
  - Rate and pitch controls.
  - Pause insertion and SSML-like tags.
  - Export to WAV and MP3.
  - Progress bar during synthesis.
- UI files live in ./webui. They use plain HTML, CSS, and small JS.
- To use Web UI:
  - Start the server.
  - Visit http://localhost:8000/ui
- The UI uses the /api/synthesize endpoint to request audio.
- It supports streaming. The UI plays audio as chunks arrive.

API reference
- All endpoints live under /api
- Health
  - GET /api/health
  - Returns server status, device, model name, and uptime.
- Info
  - GET /api/info
  - Returns model details and available voices.
- Synthesize (direct)
  - POST /api/synthesize
  - Body:
    - text: string
    - voice: optional voice name
    - sample_rate: optional int
    - format: wav | mp3 (default wav)
    - ssml: optional boolean (treat input as SSML)
    - chunk_size: optional int (override chunking)
  - Returns:
    - binary audio stream or file attachment
- Stream (server-sent events or chunked)
  - POST /api/stream
  - Accepts same payload as synthesize.
  - Returns streaming audio frames suitable for progressive playback.
- OpenAI Proxy (drop-in)
  - POST /v1/audio/speech
  - Matches a subset of OpenAI's speech endpoint for compatibility.
  - Use OPENAI_KEY or local API key to gate access.
- Model management
  - POST /api/model/load
  - Body: { model: "repo_or_path", device: "cpu|cuda" }
  - Returns load status.
- Example responses use JSON for status and binary for audio. The server sets Content-Type correctly.

Large-text / audiobook mode
- Kitten-TTS-Server supports large inputs for audiobook and long-form narration.
- Strategy:
  - Sentence segmentation using punctuation and heuristics.
  - Keep chunk size under MAX_TEXT_CHUNK to stay within memory and speed limits.
  - Insert short pauses between chunks to mimic breaths and chapter breaks.
  - Support for SSML tags and pause markers.
- Controls:
  - max_chunk_chars: maximum characters per chunk.
  - overlap_chars: number of characters to maintain overlap between chunks to preserve context (default 40).
  - pause_between_chunks_ms: pause duration inserted between chunks (default 180 ms).
- Use the audiobook endpoint:
  - POST /api/audiobook
  - Body:
    - title: optional
    - text: the full long form text
    - voice: voice preset
    - chapter_split: boolean (true to auto split by chapters using headings)
  - The server returns a single combined audio file or zip of per-chapter files.
- This mode uses streaming and incremental synthesis to keep memory use low.

GPU tips and CUDA setup
- Use GPU to accelerate model inference. The server supports PyTorch GPU builds.
- Install a CUDA-compatible PyTorch wheel for your CUDA version.
  - Check https://pytorch.org for the correct index.
- Validate CUDA:
  - python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
- Use KITTEN_DEVICE=cuda to force GPU.
- For low VRAM GPUs:
  - Use model quantization or mixed precision (fp16).
  - Set half precision in config: USE_FP16=true
  - Use dynamic batching to avoid OOM.
- For multi-GPU:
  - Use model sharding across GPUs or use a single device per container and scale horizontally.
- Common issues:
  - Mismatched PyTorch and CUDA versions cause failures.
  - Driver mismatch can show CUDA errors. Install NVIDIA drivers that match toolkit support.

Model management and Hugging Face
- The server can load models from a local path or from a Hugging Face repo id.
- Supported formats:
  - PyTorch .pt or .bin model files with codec wrappers.
  - Transformers-style repos that expose generate() compatible APIs.
- To load from Hugging Face:
  - Set KITTEN_MODEL_PATH to the repo id: e.g., facebook/kitten-tts-small
  - The server downloads the model on first run and caches it under ~/.cache/kitten-tts
- Caching:
  - Use HF_HOME or set CACHE_DIR to change cache location.
- Custom models:
  - Drop a model folder into ./models and call the load endpoint with its folder name.
- Model conversion:
  - Scripts in tools/ convert other model formats into a compatible pack. Use the release installer to run conversion if needed.

OpenAI-style endpoint proxy
- The server exposes an OpenAI-compatible route for simpler integration.
- Example:
  - POST /v1/audio/speech
  - Body: { "input": "Hello world", "voice": "kitten", "format": "wav" }
  - Authorization: Bearer <KEY>
- Use this to replace remote TTS calls with local calls in apps that expect an OpenAI-like API.

Streaming audio and real-time
- The streaming route sends audio in small chunks.
- The client can begin playback before the entire synthesis finishes.
- The server supports:
  - WebSocket streaming
  - Server-sent events (SSE)
  - HTTP chunked responses
- Use the stream endpoint when you need low latency or progressive playback.
- The server also supports a push model where audio frames are posted to a websocket consumer.

Security and access
- Bind to localhost during development. Set HOST to 127.0.0.1
- Use API keys:
  - Set an API key via OPENAI_KEY or KITTEN_API_KEY.
  - The server checks Authorization headers.
- Use TLS:
  - Use a reverse proxy like Nginx with TLS termination.
- CORS:
  - Set CORS_ORIGINS to limit allowed browser origins.
- Rate limit:
  - Use a gateway or set up per-key rate limiting if you need to throttle usage.

Logging and metrics
- The server logs to stdout in JSON format by default.
- Set LOG_LEVEL to adjust verbosity.
- Metrics:
  - Integration with Prometheus via /metrics endpoint.
  - Expose request latency, synthesize time, and memory usage.
- Store logs with a log agent or to a file for audits.

Benchmarks and perf tuning
- Baseline on CPU (Intel i7, 4 threads):
  - ~0.8x real time for short phrases (system dependent)
- Baseline on GPU (RTX 3060, 12GB):
  - ~0.05x real time for short phrases
- Memory tips:
  - Lower sample rate for lower memory at cost of fidelity.
  - Use batch synthesis for multiple short phrases.
- Tuning:
  - Increase workers in uvicorn for parallel HTTP handling.
  - Use pinned memory and preloaded models to cut cold-start time.

Examples

curl synthesize (download WAV)
- curl example:
  - curl -X POST "http://localhost:8000/api/synthesize" \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello from Kitten TTS","voice":"kitten","format":"wav"}' \
    --output hello.wav

Python example (requests)
- import requests
- payload = {"text":"Hello world from Kitten TTS","voice":"kitten","format":"wav"}
- r = requests.post("http://localhost:8000/api/synthesize", json=payload, stream=True)
- with open("out.wav","wb") as f:
-     for chunk in r.iter_content(chunk_size=8192):
-         if chunk:
-             f.write(chunk)

Streaming with WebSocket (JS snippet)
- const ws = new WebSocket("ws://localhost:8000/ws/stream");
- ws.onopen = () => ws.send(JSON.stringify({text: "A streamed line", voice:"kitten"}));
- ws.onmessage = (evt) => {
-   // handle base64 audio chunks
- };

CLI client
- The repo includes a simple CLI client in scripts/kitten-cli
- Usage:
  - kitten-cli synth "Text goes here" --voice kitten --out file.wav
- The CLI supports batch mode for many files:
  - kitten-cli batch --input texts/ --out audio/

Systemd and service deploy
- Example unit file (kitten-tts.service)
  - [Unit]
    Description=Kitten TTS Server
    After=network.target
  - [Service]
    User=kitten
    WorkingDirectory=/opt/kitten-tts
    EnvironmentFile=/opt/kitten-tts/.env
    ExecStart=/opt/kitten-tts/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
    Restart=on-failure
  - [Install]
    WantedBy=multi-user.target
- Place the unit file in /etc/systemd/system and enable it:
  - sudo systemctl daemon-reload
  - sudo systemctl enable --now kitten-tts

Troubleshooting
- Server fails to start on CUDA:
  - Check torch and CUDA versions.
  - Validate GPU visibility with nvidia-smi.
  - If driver missing, install the correct NVIDIA driver.
- Audio artifacts:
  - Check sample rate mismatch between model and output.
  - Use provided resampler if needed.
- High memory:
  - Lower chunk size and sample rate.
  - Use fp16 if supported.
- Model download failure:
  - Set HF token via HUGGINGFACE_TOKEN or use offline model path.
- Port conflicts:
  - Change PORT or stop other service using the port.

Contributing
- The code aims to be modular and simple.
- To contribute:
  - Fork the repo.
  - Create a feature branch.
  - Run tests: pytest -q
  - Open a PR with clear description and tests.
- Code style:
  - Follow PEP8.
  - Keep functions small and focused.
  - Document new features in README and add tests.

License
- The project uses an open license. See the LICENSE file in the repo for details. Use it in a manner consistent with model licenses and third-party terms.

Acknowledgements
- Kitten TTS model authors and the Hugging Face community.
- FastAPI and Uvicorn for the server core.
- PyTorch and CUDA for GPU compute.
- Open source contributors and test users.

Assets and images
- Kitten image courtesy of Unsplash (public images). Use images that fit your license.
- Waveform or audio icon from common icon packs.
- Replace web assets in ./webui/static.

Advanced topics

Custom voice presets
- Define voices in voices.yaml.
- Each voice includes:
  - model: path or HF id
  - sample_rate: int
  - prosody: dict with rate and pitch defaults
  - postproc: optional filters to apply to the audio stream
- Load a voice:
  - POST /api/voice/load { "voice":"soft_kitten" }

Batch synthesis and concurrency
- Batch mode accepts JSON list of texts.
- It returns a zip of generated files.
- Use worker pools for parallel tasks.
- Example:
  - POST /api/synthesize/batch with body [ {text:"a",voice:"v1"}, {...} ]

SSML and pause control
- The server accepts minimal SSML tags:
  - <break time="500ms"/>
  - <emphasis level="moderate">text</emphasis>
- SSML is translated to prosody markers in the model input.
- Limitations:
  - Not all SSML features work. Use tags that map to pause and pitch.

Export formats and codecs
- Formats supported:
  - WAV (PCM)
  - MP3 (via pydub or lame)
  - OGG (via ffmpeg)
- To enable MP3, install ffmpeg and pydub.
- Default sample rate is 22050 Hz. Set SAMPLE_RATE to change.

Integration patterns
- Replace cloud TTS endpoints in existing apps by switching the URL to this server.
- Use the OpenAI-like endpoint to minimize code changes.
- For mobile apps, stream audio chunks for low latency.

Model size and runtime options
- Small Kitten TTS models fit in low-memory environments.
- Use dynamic quantization for CPU-only devices.
- For embedded or ARM devices, cross-compile or use CPU builds and smaller model weights.

Testing
- Unit tests live in tests/.
- Run:
  - pytest -q
- Add tests for endpoints, model loader, and chunker.

Release notes and installers
- Binary releases and installer scripts are on the Releases page. Download the release bundle and run the provided installer in the package.
- Visit the releases page to get the latest stable binary or installer:
  - https://github.com/SamThinks-Com/Kitten-TTS-Server/releases

Community and support
- Open issues for bugs and feature requests.
- Submit PRs for fixes and improvements.
- Use discussions to ask questions or share use cases.

Extending the server
- Add new model backends under app/models/.
- Implement a new API route in app/api/.
- Add voice presets in voices.yaml and precompute prosody settings.

Packaging and distribution
- Packaging scripts build wheel and Docker images.
- The release installer bundles the wheel and a minimal runtime.
- Use the release installer if you want a guided install.

Privacy and offline use
- Self-hosting keeps data local.
- Disable any external network calls by setting OFFLINE_MODE=true.
- If the model downloads are a concern, pre-download and place models under ./models.

Maintenance tips
- Keep PyTorch and CUDA in sync.
- Monitor GPU driver updates.
- Periodically prune cache to free disk space.

Developer notes
- Code is modular with clear boundaries:
  - app/main.py: server entry
  - app/models/: model wrappers
  - app/api/: endpoints
  - webui/: static UI assets
  - tools/: helper scripts
- Follow tests and CI for contributions.

Release link (again)
- Download the packaged installer or binary from the Releases page and run the installer found in the downloaded package:
  - https://github.com/SamThinks-Com/Kitten-TTS-Server/releases

Examples of real workflows

1) Local dev with CPU
- git clone ...
- python -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- uvicorn app.main:app --reload

2) GPU dev
- Install CUDA and matching torch wheel.
- Set KITTEN_DEVICE=cuda
- pip install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 8000

3) Production with Docker and GPU
- docker build -t kitten-tts-server .
- docker run --gpus all -p 8000:8000 kitten-tts-server

4) Audiobook generation
- POST /api/audiobook with the whole book text.
- Receive a zip with chapter WAVs or a single combined track.

Caveats and compatibility
- The server works best with compatible model formats. If you load a model with a nonstandard interface, provide a wrapper in app/models/.
- Keep models and server in separate upgrade cycles to reduce breakage.

Appendix: Helpful commands
- Validate environment:
  - python -c "import sys,torch; print(sys.version, torch.__version__)"
- Check disk:
  - df -h
- Check GPU:
  - nvidia-smi

Files of interest in the repo
- app/ - main application code
- webui/ - UI assets
- scripts/ - helper scripts and CLI
- tools/ - conversion and packaging tools
- tests/ - unit and integration tests
- requirements.txt - Python deps
- docker/ - Dockerfile and compose manifests
- voices.yaml - voice presets
- LICENSE - license text

Images and icons used here
- Kitten photo from Unsplash.
- Waveform images can be generated from sample audio using ffmpeg or use public domain icons.

End of README content.