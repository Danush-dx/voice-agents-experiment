# Quick Setup Guide

## 1. Create Configuration File

Copy the example configuration:
```bash
cp .env.example .env
```

## 2. Edit Configuration

Open `.env` in your editor and set:

```bash
# Required
HF_TOKEN=your_huggingface_token_here
BASE_URL=https://your-ngrok-url.ngrok-free.app

# Optional (defaults are fine for local dev)
PORT=8080
ASR_MODEL_SIZE=large-v3
LOG_LEVEL=INFO
```

### Getting HuggingFace Token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access)
3. Accept PyAnnote terms: https://huggingface.co/pyannote/speaker-diarization-3.1
4. Copy token to `.env` file

### Getting ngrok URL:
```bash
# Start ngrok in a separate terminal
ngrok http 8080
```
Copy the HTTPS URL (e.g., `https://xxxx.ngrok-free.app`) and paste into `.env`

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Server

```bash
./start_telephony.sh
```

That's it! The script will:
- Validate your configuration
- Display all settings
- Start the telephony server
- Show you the webhook URL for Vobiz

## 5. Configure Vobiz

In your Vobiz dashboard:
1. Go to Phone Numbers → Select your number
2. Set Webhook URL to: `{BASE_URL}/api/telephony/answer`
3. Save and test by calling your number

## Configuration Options

All options can be set in `.env` file:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HF_TOKEN` | HuggingFace API token | - | Yes (for PyAnnote) |
| `BASE_URL` | Public URL for webhooks | - | Yes |
| `HOST` | Server bind host | 0.0.0.0 | No |
| `PORT` | Server port | 8080 | No |
| `VAD_TYPE` | VAD engine | pyannote | No |
| `ASR_TYPE` | ASR engine | faster_whisper | No |
| `ASR_MODEL_SIZE` | Whisper model size | large-v3 | No |
| `ASR_DEVICE` | cpu or cuda | cpu | No |
| `ASR_COMPUTE_TYPE` | Precision | int8 | No |
| `LOG_LEVEL` | Logging level | INFO | No |

## Troubleshooting

### "ERROR: .env file not found"
Run: `cp .env.example .env` and edit the file

### "ERROR: Missing required configuration"
Edit `.env` and set `HF_TOKEN` and `BASE_URL`

### ngrok connection issues
- Ensure ngrok is running: `ngrok http 8080`
- Use HTTPS URL (not HTTP)
- Update `BASE_URL` in `.env` when ngrok restarts

### Port already in use
Change `PORT` in `.env` to a different port (e.g., 8000)
Also update ngrok: `ngrok http 8000`

## Advanced Usage

You can override any config with command-line args:
```bash
python -m src.telephony_main --port 8000 --asr-args '{"model_size": "medium"}'
```

View all options:
```bash
python -m src.telephony_main --help
```
