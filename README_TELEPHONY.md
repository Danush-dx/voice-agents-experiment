# VoiceStreamAI Telephony Setup Guide

## Quick Start

### 1. Get HuggingFace Token
- Visit: https://huggingface.co/settings/tokens
- Create a new token (read access)
- Accept PyAnnote terms: https://huggingface.co/pyannote/speaker-diarization-3.1

### 2. Start ngrok Tunnel
In Terminal 1:
```bash
ngrok http 8080
```

Copy the HTTPS URL (e.g., `https://xxxx-xx-xx-xx-xx.ngrok-free.app`)

### 3. Start Telephony Server
In Terminal 2:
```bash
export HF_TOKEN=your_huggingface_token_here
export BASE_URL=https://xxxx-xx-xx-xx-xx.ngrok-free.app

./start_telephony.sh
```

Or as a one-liner:
```bash
HF_TOKEN=your_token BASE_URL=https://your-ngrok-url.ngrok-free.app ./start_telephony.sh
```

### 4. Configure Vobiz
1. Log into your Vobiz dashboard
2. Go to Phone Numbers → Select your number
3. Set Webhook URL to: `https://your-ngrok-url.ngrok-free.app/api/telephony/answer`
4. Save configuration

### 5. Test the Setup
- Call your Vobiz phone number
- Speak into the phone
- Watch the terminal for transcription output

## Endpoints

- **Health Check**: `GET /`
- **Answer Webhook**: `POST /api/telephony/answer` (for Vobiz)
- **Audio Stream**: `WS /api/telephony/stream` (WebSocket for audio)

## Troubleshooting

### "ERROR: Please set your HuggingFace token"
Set the HF_TOKEN environment variable:
```bash
export HF_TOKEN=your_token_here
```

### "ERROR: Please set BASE_URL"
Make sure ngrok is running and copy its HTTPS URL:
```bash
export BASE_URL=https://xxxx.ngrok-free.app
```

### ngrok connection issues
- Check ngrok is running in a separate terminal
- Verify the URL is HTTPS (not HTTP)
- Make sure port 8080 is not already in use

### Audio quality issues
- Check your internet connection
- Verify Vobiz is sending µ-law encoded audio
- Check server logs for conversion errors

### Model download taking long
First run will download:
- PyAnnote VAD models (~500MB)
- Faster Whisper large-v3 model (~3GB)
- These are cached for future use

## Manual Start (without script)

```bash
python -m src.telephony_main \
    --host 0.0.0.0 \
    --port 8080 \
    --base-url https://your-ngrok-url.ngrok-free.app \
    --vad-type pyannote \
    --vad-args '{"auth_token": "your_hf_token"}' \
    --asr-type faster_whisper \
    --asr-args '{"model_size": "large-v3", "device": "cpu", "compute_type": "int8"}'
```

## Architecture

```
Phone Call → Vobiz → ngrok → FastAPI (8080)
                                ↓
                          Audio Converter
                          (µ-law 8kHz → PCM 16kHz)
                                ↓
                          VAD Pipeline (PyAnnote)
                                ↓
                          ASR Pipeline (Faster Whisper)
                                ↓
                          Transcription JSON → Vobiz
```

## Production Deployment

For production, replace ngrok with:
- A server with public IP/domain
- SSL certificates (Let's Encrypt)
- Use `--ssl-certfile` and `--ssl-keyfile` flags

Example:
```bash
python -m src.telephony_main \
    --host 0.0.0.0 \
    --port 443 \
    --base-url wss://your-domain.com \
    --ssl-certfile /path/to/cert.pem \
    --ssl-keyfile /path/to/key.pem \
    --vad-type pyannote \
    --vad-args '{"auth_token": "token"}' \
    --asr-type faster_whisper \
    --asr-args '{"model_size": "large-v3"}'
```
