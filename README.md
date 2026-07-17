# HEARING-MODEL

AI server for **Hear:ing** — a sound-awareness assistant for deaf and hard-of-hearing users.

The server receives ambient audio streamed from the wearable ESP32 device, classifies the sound using Google YAMNet, maps the raw YAMNet label into the service's Korean sound catalog, and forwards the classification result and sound direction to the Hear:ing backend.

The AI server is responsible only for **audio classification**. User mode matching, per-sound filtering, notification storage, push delivery, and final vibration decisions are handled by `HEARING-BE`.

Capstone project, Duksung Women's University, 2026.

Sibling repositories:

- `HEARING-FE` — React PWA
- `HEARING-BE` — FastAPI backend
- `HEARING-MODEL` — YAMNet AI server

---

## Architecture

```text
ESP32 wearable
    │
    │ WebSocket binary packet
    │ [direction: 1 byte][padding: 3 bytes][PCM int16 audio]
    ▼
HEARING-MODEL
    │
    ├─ Parse direction header
    ├─ Extract PCM int16 audio
    ├─ Normalize audio waveform
    ├─ Run YAMNet inference
    ├─ Apply Korean category mapping
    ├─ Remove low-confidence and unmapped results
    └─ Select top sound results
    │
    │ REST API
    ▼
HEARING-BE
    │
    ├─ Match active mode and enabled sounds
    ├─ Store notification
    ├─ Push alert to web app
    └─ Send vibration command to wearable
```

---

## Audio classification flow

1. The ESP32 connects to the AI server through WebSocket.
2. The wearable monitors ambient sound through its microphones.
3. The ESP32 sends a binary audio packet containing direction information and PCM audio.
4. The AI server reads the first byte as the sound direction.
5. The following three padding bytes are skipped.
6. The remaining bytes are interpreted as signed 16-bit PCM audio.
7. The waveform is normalized and passed to YAMNet.
8. YAMNet produces scores for 521 AudioSet sound classes.
9. The AI server maps the YAMNet class names to the Hear:ing Korean sound catalog.
10. Results below the minimum score or mapped to `기타` are removed.
11. Duplicate results with the same category and sound name are merged.
12. Up to three final results are returned through WebSocket.
13. The highest-ranked result is forwarded to `HEARING-BE`.

---

## WebSocket audio packet contract

The ESP32 sends each audio message in the following binary format:

```text
[1 byte: direction]
[3 bytes: padding]
[remaining bytes: PCM signed int16 little-endian audio]
```

### Direction values

| Value | Direction |
|---:|---|
| `0` | `FRONT` |
| `1` | `BACK` |
| `2` | `LEFT` |
| `3` | `RIGHT` |
| `4` | `UNKNOWN` |

Example packet:

```text
02 00 00 00 [PCM audio bytes...]
```

The packet above indicates that the sound direction is `LEFT`.

Invalid direction values are converted to `UNKNOWN`.

The PCM payload must:

- contain at least one audio sample;
- have an even byte length;
- use signed 16-bit little-endian samples;
- use a 16 kHz sample rate.

---

## Classification output

The classifier returns up to three mapped sound results.

Example:

```json
{
  "status": "success",
  "direction": "LEFT",
  "direction_value": 2,
  "top_sounds": [
    {
      "category": "교통",
      "name": "경적",
      "yamnet_class": "Vehicle horn, car horn, honking",
      "confidence": 0.8231,
      "percent": 62.48
    },
    {
      "category": "교통",
      "name": "차량 주행음",
      "yamnet_class": "Vehicle",
      "confidence": 0.3014,
      "percent": 22.88
    },
    {
      "category": "생활음",
      "name": "소음",
      "yamnet_class": "Environmental noise",
      "confidence": 0.1931,
      "percent": 14.64
    }
  ]
}
```

`percent` is normalized across the returned results so that the displayed percentages total approximately 100%.

The backend receives the highest-ranked result.

Example backend payload:

```json
{
  "sound_category": "교통",
  "sound_name": "경적",
  "confidence": 0.8231,
  "direction": "LEFT",
  "detected_at": "2026-07-17T10:30:00+00:00"
}
```

---

## Sound mapping

YAMNet returns English AudioSet class names such as:

```text
Siren
Fire alarm
Vehicle horn, car horn, honking
Baby cry, infant cry
Vacuum cleaner
Rain
```

The AI server converts these labels into the Korean category and sound names used by the frontend and backend.

Example:

```python
"Fire alarm": ("긴급", "화재 경보")
"Siren": ("긴급", "사이렌")
"Vehicle horn, car horn, honking": ("교통", "경적")
"Baby cry, infant cry": ("사람", "아기 울음")
"Vacuum cleaner": ("생활음", "청소기")
"Rain": ("자연", "비")
```

Labels missing from the project mapping are returned as:

```python
("기타", "기타")
```

Results mapped to `기타` are excluded from the final output.

The Korean category and sound-name strings are part of the contract shared with `HEARING-BE` and `HEARING-FE`. Mapping changes must therefore be coordinated across repositories.

---

## Classification rules

The classifier applies the following post-processing rules:

| Setting | Default | Description |
|---|---:|---|
| `MIN_SCORE` | `0.1` | Discards predictions below this score |
| `RAW_TOP_K` | `20` | Examines the top 20 raw YAMNet classes |
| `RESULT_LIMIT` | `3` | Returns up to three final mapped results |
| `ALERT_THRESHOLD` | Environment setting | Minimum confidence required for backend delivery |
| `COOLDOWN_SECONDS` | Environment setting | Suppresses repeated detections for the same sound and direction |

Results with the same `(category, name)` pair are treated as one service-level sound. Only the highest-confidence YAMNet result is retained.

Example:

```text
Vehicle
Car
Motor vehicle (road)
```

These classes may all map to:

```text
교통 / 차량 주행음
```

Only the highest-scoring result is returned.

---

## Tech stack

- Python 3.11
- FastAPI
- Uvicorn
- TensorFlow
- TensorFlow Hub
- YAMNet
- NumPy
- HTTPX
- PyJWT
- WebSocket
- Docker
- AWS Lightsail

---

## Project layout

```text
app/
  api/
    audio.py              # ESP32 audio WebSocket endpoint

  core/
    config.py             # Environment and runtime settings
    direction.py          # Direction enum and audio-packet parser

  data/
    category_map.py       # YAMNet label to Korean sound mapping

  services/
    classifier.py         # YAMNet loading, inference and post-processing
    backend_client.py     # Detection request to HEARING-BE

  main.py                 # FastAPI application initialization

example.env               # Example environment variables
requirements.txt          # Python dependencies
run.py                    # Local Uvicorn entry point
Dockerfile                # Container image definition
```

---

## Getting started

### Prerequisites

- Python 3.11
- Git
- Internet access during the first execution
- A running `HEARING-BE` server for backend integration

YAMNet is downloaded from TensorFlow Hub when it is not already cached.

### Clone the repository

```powershell
git clone https://github.com/2026-DSWU-Hearing/HEARING-MODEL.git
cd HEARING-MODEL
```

### Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment variables

Copy the example configuration:

```powershell
Copy-Item example.env .env
```

Example configuration:

```env
APP_NAME=HEARING-MODEL
HOST=0.0.0.0
PORT=8001

BACKEND_URL=http://localhost:8000
JWT_SECRET=replace-with-the-same-secret-used-by-hearing-be
DEVICE_ID=1

ALERT_THRESHOLD=0.1
COOLDOWN_SECONDS=5
```

`JWT_SECRET` must match the value configured in `HEARING-BE`.

The AI server uses a token containing an `ai-server` source claim when sending detection results to the backend.

Example token payload:

```json
{
  "sub": "1",
  "source": "ai-server",
  "type": "access"
}
```

### Run the server

```powershell
py -3.11 run.py
```

Alternatively:

```powershell
py -3.11 -m uvicorn app.main:app --reload --port 8001
```

The API server runs at:

```text
http://127.0.0.1:8001
```

Interactive API documentation:

```text
http://127.0.0.1:8001/docs
```

---

## WebSocket connection

The wearable connects to the AI server through the configured WebSocket endpoint.

Example:

```text
ws://<AI_SERVER_HOST>:8001/ws/audio
```

After connection, the ESP32 sends binary packets using the direction-header and PCM format documented above.

### Successful classification

```json
{
  "status": "success",
  "direction": "LEFT",
  "direction_value": 2,
  "top_sounds": [
    {
      "category": "교통",
      "name": "경적",
      "yamnet_class": "Vehicle horn, car horn, honking",
      "confidence": 0.8231,
      "percent": 100.0
    }
  ]
}
```

### No mapped sound detected

```json
{
  "status": "not_detected",
  "direction": "UNKNOWN",
  "direction_value": 4,
  "top_sounds": []
}
```

### Invalid packet

```json
{
  "status": "error",
  "message": "PCM int16 데이터 길이는 짝수여야 합니다."
}
```

---

## Backend integration

After classification, the AI server sends the highest-confidence sound result to:

```text
POST /devices/{device_id}/detections
```

Example request body:

```json
{
  "sound_category": "교통",
  "sound_name": "경적",
  "confidence": 0.8231,
  "direction": "LEFT",
  "detected_at": "2026-07-17T10:30:00+00:00"
}
```

The AI server does not determine whether the user should receive an alert.

`HEARING-BE` performs the final processing:

```text
classified result received
→ find the registered device and owning user
→ check the user's active mode
→ check the per-sound enabled setting
→ check do-not-disturb
→ save the notification
→ send an FCM push
→ broadcast through the user WebSocket
→ command the wearable to vibrate
```

When no active-mode match exists, the backend silently ignores the detection and stores nothing.

---

## Running with Docker

Build the image:

```powershell
docker build -t hearing-model .
```

Run the container:

```powershell
docker run --env-file .env -p 8001:8001 hearing-model
```

The first startup may take longer because TensorFlow Hub must download and cache YAMNet.

---

## TensorFlow Hub cache

YAMNet is loaded from:

```text
https://tfhub.dev/google/yamnet/1
```

TensorFlow Hub stores the downloaded model in a local cache.

If the cache is incomplete, startup may fail with an error similar to:

```text
contains neither 'saved_model.pb' nor 'saved_model.pbtxt'
```

Remove the broken cache:

```powershell
Remove-Item -Recurse -Force "$env:TEMP\tfhub_modules"
```

Then restart the server so that YAMNet is downloaded again.

A custom cache directory can also be configured before importing TensorFlow Hub:

```python
import os

os.environ["TFHUB_CACHE_DIR"] = ".tfhub_cache"
```

Do not commit TensorFlow Hub cache files.

Recommended `.gitignore` entry:

```gitignore
.tfhub_cache/
```

---

## Development notes

### Audio format

`classifier.py` expects pure PCM audio bytes.

Do not pass the complete ESP32 packet directly to the classifier:

```python
# Incorrect
result = classifier.classify(packet)
```

Parse the packet first:

```python
direction_value, direction_name, pcm_audio = parse_audio_packet(packet)
result = classifier.classify(pcm_audio)
```

Otherwise, the four-byte direction header will be interpreted as audio samples and may affect classification.

### Direction handling

Direction is generated by the wearable hardware. It is not inferred by YAMNet.

The AI server:

1. parses the direction byte;
2. attaches the direction to the classification output;
3. forwards the direction to the backend.

### Model responsibility

YAMNet only classifies the sound.

The AI server reports:

```text
sound category
sound name
confidence
direction
detection time
```

The backend performs:

```text
active-mode matching
per-sound toggle checking
do-not-disturb checking
notification storage
FCM push delivery
in-app WebSocket broadcasting
wearable vibration commands
```

---

## Planned hardware AI

The current server uses the complete YAMNet model for detailed sound classification.

A separate lightweight hardware AI project is planned for the ESP32-S3:

```text
ambient audio
→ small emergency-versus-normal classifier
→ emergency: perform an immediate vibration response
→ send audio or detection information to the server
→ server-side YAMNet performs detailed classification
```

The ESP32-S3 will not run the complete 521-class YAMNet model.

Instead, YAMNet will be used as a teacher or feature extractor to train a smaller CNN model suitable for TensorFlow Lite Micro.

The planned training flow is:

```text
ESC-50 audio dataset
→ YAMNet-based feature extraction or reference classification
→ emergency / normal dataset construction
→ small CNN training
→ INT8 quantization
→ TensorFlow Lite Micro deployment
→ ESP32-S3 inference
```

The server-side YAMNet pipeline remains in this repository.

---

## Known limitations

- Classification accuracy depends on microphone quality, volume, distance, background noise, and recording environment.
- The classifier assumes 16 kHz signed PCM int16 audio.
- Only labels included in `category_map.py` can appear in the final result.
- Valid YAMNet labels missing from the project mapping are discarded as `기타`.
- Direction accuracy depends on the ESP32 microphone and direction-detection implementation.
- Up to three results may be returned through WebSocket, but only the highest-ranked result is sent to the backend.
- CPU-only inference may have increased latency during concurrent requests.
- The first startup requires a YAMNet download unless a valid cache already exists.
- The AI server and backend currently share authentication configuration.
- The Korean sound mapping must remain synchronized with the backend sound catalog.

---

## Deployment notes

Current target environment:

```text
AWS Lightsail
4 GB RAM
Docker container
Python 3.11
FastAPI
TensorFlow
YAMNet
```

Deployment checklist:

- Set the production `BACKEND_URL`.
- Use the same strong `JWT_SECRET` as `HEARING-BE`.
- Configure the correct registered `DEVICE_ID`.
- Use `wss://` and `https://` in production.
- Ensure sufficient disk space for TensorFlow and the YAMNet cache.
- Restrict inbound ports to required API and WebSocket endpoints.
- Verify that backend and AI-server sound mappings are synchronized.
- Do not commit `.env`, model caches, uploaded audio or Python cache files.

Recommended `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.tfhub_cache/
app/uploads/
```

---

## Repository responsibilities

| Repository | Responsibility |
|---|---|
| `HEARING-FE` | User interface, mode configuration, sound settings, alert history and account settings |
| `HEARING-BE` | Authentication, users, modes, devices, final alert decisions, notifications, push messages and vibration routing |
| `HEARING-MODEL` | PCM audio reception, direction parsing, YAMNet inference, sound mapping and classified-result delivery |
