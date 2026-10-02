# Whisper-WebUI
A Gradio-based browser interface for [Whisper](https://github.com/openai/whisper). You can use it as an Easy Subtitle Generator!

![screen](https://github.com/user-attachments/assets/caea3afd-a73c-40af-a347-8d57914b1d0f)



## Notebook
If you wish to try this on Colab, you can do it in [here](https://colab.research.google.com/github/jhj0517/Whisper-WebUI/blob/master/notebook/whisper-webui.ipynb)!

# Feature
- Select the Whisper implementation you want to use between :
   - [openai/whisper](https://github.com/openai/whisper)
   - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) (used by default)
   - [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- Generate subtitles from various sources, including :
  - Files
  - Youtube
  - Microphone
- Currently supported subtitle formats : 
  - SRT
  - WebVTT
  - txt ( only text file without timeline )
- Speech to Text Translation 
  - From other languages to English. ( This is Whisper's end-to-end speech-to-text translation feature )
- Text to Text Translation
  - Translate subtitle files using Facebook NLLB models
  - Translate subtitle files using DeepL API
- Pre-processing audio input with [Silero VAD](https://github.com/snakers4/silero-vad).
- Pre-processing audio input to separate BGM with [UVR](https://github.com/Anjok07/ultimatevocalremovergui). 
- Post-processing with speaker diarization using the [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1) model.
   - To download the pyannote model, you need to have a Huggingface token and manually accept their terms in the pages below.
      1. https://huggingface.co/pyannote/speaker-diarization-3.1
      2. https://huggingface.co/pyannote/segmentation-3.0

### Pipeline Diagram
![Transcription Pipeline](https://github.com/user-attachments/assets/1d8c63ac-72a4-4a0b-9db0-e03695dcf088)

# Installation and Running

- ## Running with Pinokio

The app is able to run with [Pinokio](https://github.com/pinokiocomputer/pinokio).

1. Install [Pinokio Software](https://program.pinokio.computer/#/?id=install).
2. Open the software and search for Whisper-WebUI and install it.
3. Start the Whisper-WebUI and connect to the `http://localhost:7860`.

- ## Running with Docker 

1. Install and launch [Docker-Desktop](https://www.docker.com/products/docker-desktop/).

2. Git clone the repository

```sh
git clone https://github.com/jhj0517/Whisper-WebUI.git
```

3. Build the image ( Image is about 7GB~ )

```sh
docker compose build 
```

4. Run the container 

```sh
docker compose up
```

5. Connect to the WebUI with your browser at `http://localhost:7860`

ElevenLabs is the default provider for uploaded files, with speaker diarization enabled
by default. Put one or more keys in the repository-root `.env`; keys are tried in
numeric order and are never sent to the browser:

```dotenv
ELEVEN_LABS_KEY_1=your-key
ELEVEN_LABS_KEY_2=another-key
```

Docker Compose loads this `.env` file into the container. Do not commit it. When every
configured key is unavailable or ElevenLabs remains unavailable after bounded retries,
the complete source file is transcribed by Whisper. You can also select Whisper directly
under the `Транскрибация` button. Files longer than the safe 8-hour request target are
converted to sequential mono 16 kHz FLAC chunks with a 2-second overlap. The overlap is
billed twice by ElevenLabs, and transient retries may also consume credits.
The implementation stays below ElevenLabs' documented 10-hour duration limit and uses
3 GB as a conservative hard upload ceiling (their current documentation is inconsistent
between 3 GB and 5 GB).
For diarized chunked files, labels use `SPEAKER_00|text`; because each chunk is
diarized independently, the same physical voice cannot be guaranteed to retain the
same label across every boundary without a separate speaker-matching model.

If needed, update the [`docker-compose.yaml`](https://github.com/jhj0517/Whisper-WebUI/blob/master/docker-compose.yaml) to match your environment.

### Private HTTPS and coordinated cold apply

The retained portal frame uses `https://transcriber.quasilegend.ru/`. Uvicorn must
trust only the verified private gateway hop to preserve HTTPS redirects. Compose
retains loopback trust and adds `WHISPER_GATEWAY_PROXY_IP` from the ignored `.env`.
Without that selected value, only loopback is trusted. Never use `*`, a CIDR, or
an arbitrary proxy list. Verify the TCP peer through the gateway's actual Docker
network and host-published route; a container's default gateway alone is not proof.

`apply_prebuilt.sh` is the supported cold-only entrypoint. Supply exactly
`--image=whisper-webui=sha256:ID` and `--proxy-ip=PRIVATE_IPV4` from the sealed
release packet. It requires clean canonical `master`, a checkout lock, unchanged
HEAD/resolved configuration and the same expected local image ID initially and
immediately before apply. The helper overrides `WHISPER_PREBUILT_IMAGE` with that
immutable ID rather than resolving a mutable tag at apply. Normal Compose keeps
the existing image default; do not set this override for ordinary builds.
It never builds, pulls, pushes or deletes volumes, and
applies only `whisper-webui`. The resolved trust must be loopback plus that one IP.
Use `tests/test_prebuilt_deployment.py` for fake-command regression validation.

A clean release checkout must set `WHISPER_STATE_ROOT` to the existing absolute
state directory (models/configs/outputs) and preserve the existing Compose project
name. The default `.` retains ordinary local behavior. Verify all resolved bind
sources against the running container before apply; never initialize an empty
state directory or replace the working configuration with Git defaults.

Before the single coordinated window, validate the source/image provenance,
preserve those binds and protect only irreplaceable at-risk configuration/output
state, not reconstructible model caches. Stop new work and drain active transcription tasks
without cancellation. The helper's two-minute stop grace and Compose `--wait`
(running state, no application healthcheck) do not prove task drain or HTTPS
readiness. Verify redirect, API auth and browser behavior afterward without paid
inference. Existing credentials, mounted data, networks and launch mode remain.
No standalone restart is implied by this source preparation.

- ## Run Locally

### Prerequisite
To run this WebUI, you need to have `git`, `3.10 <= python <= 3.12`, `FFmpeg`.

**Edit `--extra-index-url` in the [`requirements.txt`](https://github.com/jhj0517/Whisper-WebUI/blob/master/requirements.txt) to match your device.<br>** 
By default, the WebUI assumes you're using an Nvidia GPU and **CUDA 12.8.** If you're using Intel or another CUDA version, read the [`requirements.txt`](https://github.com/jhj0517/Whisper-WebUI/blob/master/requirements.txt) and edit `--extra-index-url`.

Please follow the links below to install the necessary software:
- git : [https://git-scm.com/downloads](https://git-scm.com/downloads)
- python : [https://www.python.org/downloads/](https://www.python.org/downloads/) **`3.10 ~ 3.12` is recommended.** 
- FFmpeg :  [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- CUDA : [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

After installing FFmpeg, **make sure to add the `FFmpeg/bin` folder to your system PATH!**

### Installation Using the Script Files

1. git clone this repository
```shell
git clone https://github.com/jhj0517/Whisper-WebUI.git
```
2. Run `install.bat` or `install.sh` to install dependencies. (It will create a `venv` directory and install dependencies there.)
3. Start WebUI with `start-webui.bat` or `start-webui.sh` (It will run `python app.py` after activating the venv)

And you can also run the project with command line arguments if you like to, see [wiki](https://github.com/jhj0517/Whisper-WebUI/wiki/Command-Line-Arguments) for a guide to arguments.

# VRAM Usages
This project is integrated with [faster-whisper](https://github.com/guillaumekln/faster-whisper) by default for better VRAM usage and transcription speed.

According to faster-whisper, the efficiency of the optimized whisper model is as follows: 
| Implementation    | Precision | Beam size | Time  | Max. GPU memory | Max. CPU memory |
|-------------------|-----------|-----------|-------|-----------------|-----------------|
| openai/whisper    | fp16      | 5         | 4m30s | 11325MB         | 9439MB          |
| faster-whisper    | fp16      | 5         | 54s   | 4755MB          | 3244MB          |

If you want to use an implementation other than faster-whisper, use `--whisper_type` arg and the repository name.<br>
Read [wiki](https://github.com/jhj0517/Whisper-WebUI/wiki/Command-Line-Arguments) for more info about CLI args.

If you want to use a fine-tuned model, manually place the models in `models/Whisper/` corresponding to the implementation.

Alternatively, if you enter the huggingface repo id (e.g, [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)) in the "Model" dropdown, it will be automatically downloaded in the directory.

![image](https://github.com/user-attachments/assets/76487a46-b0a5-4154-b735-ded73b2d83d4)

# REST API
If you're interested in deploying this app as a REST API, please check out [/backend](https://github.com/jhj0517/Whisper-WebUI/tree/master/backend).

The in-process QSD adapter accepts an optional multipart `provider` field on
`POST /qsd/transcriptions`. Valid values are `elevenlabs` (the default) and `whisper`.

## TODO🗓

- [x] Add DeepL API translation
- [x] Add NLLB Model translation
- [x] Integrate with faster-whisper
- [x] Integrate with insanely-fast-whisper
- [x] Integrate with whisperX ( Only speaker diarization part )
- [x] Add background music separation pre-processing with [UVR](https://github.com/Anjok07/ultimatevocalremovergui)  
- [x] Add fast api script
- [ ] Add CLI usages
- [ ] Support real-time transcription for microphone

### Translation 🌐
Any PRs that translate the language into [translation.yaml](https://github.com/jhj0517/Whisper-WebUI/blob/master/configs/translation.yaml) would be greatly appreciated!
