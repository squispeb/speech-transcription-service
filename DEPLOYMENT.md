# Deployment Guide

This guide deploys the transcription service on a separate Linux machine with an NVIDIA GPU and exposes it securely over HTTPS with bearer-token auth.

Recommended target:

- Ubuntu 22.04 or 24.04
- NVIDIA GPU host such as the `3060 Ti` machine
- public DNS name such as `transcribe.example.com`
- `Caddy` for TLS termination and reverse proxy
- `systemd` for process supervision

## Deployment Shape

Use this layout:

- `uvicorn` runs the FastAPI app on `127.0.0.1:8000`
- `Caddy` listens on `443` and proxies to `127.0.0.1:8000`
- the app server calls `https://transcribe.example.com/transcribe`
- every request still requires `Authorization: Bearer <TRANSCRIPTION_SERVICE_TOKEN>`

Why this setup:

- the Python service stays off the public interface
- Caddy manages HTTPS certificates automatically
- `systemd` restarts the service if it crashes
- the token remains server-to-server only

## 1. Prepare The GPU Host

Update the machine:

```bash
sudo apt update && sudo apt upgrade -y
```

Install base packages:

```bash
sudo apt install -y ffmpeg curl git build-essential caddy
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell or export the path for the current session:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Make sure NVIDIA drivers and CUDA-compatible PyTorch support are already working on that machine before continuing.

Sanity check:

```bash
nvidia-smi
```

## 2. Create A Service User

Use a dedicated account instead of running the service as your personal user:

```bash
sudo useradd --create-home --shell /bin/bash transcribe
```

## 3. Copy The Project To The Host

Clone or copy the repository onto the machine. Example:

```bash
sudo -u transcribe git clone <your-repo-url> /home/transcribe/transcription-service
```

If the service lives in its own repo already, clone that repo directly.

## 4. Configure The Service

Switch to the service user:

```bash
sudo -u transcribe -H bash
cd /home/transcribe/transcription-service
```

Create the env file:

```bash
cp .env.example .env
```

Recommended production values:

```env
TRANSCRIPTION_SERVICE_TOKEN=<generate-a-long-random-token>
PARAKEET_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v3
LANGID_MODEL_NAME=langid_ambernet
MODEL_DEVICE=cuda
MAX_UPLOAD_BYTES=10485760
MAX_AUDIO_SECONDS=120
TRANSCRIPTION_TIMEOUT_SECONDS=30
LANGID_CONFIDENCE_THRESHOLD=0.55
LANGID_MIN_MARGIN=0.15
FFMPEG_BINARY=ffmpeg
TEMP_DIRECTORY=
```

Generate a strong token with one of these:

```bash
openssl rand -hex 32
```

or:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Important:

- never commit `.env`
- use a long random token, not `replace-me`
- the same token must also be configured in the main app server

## 5. Install Python Dependencies

From the service directory:

```bash
uv sync --extra inference
```

This creates `.venv/` and installs FastAPI, NeMo, Torch, and the rest of the runtime dependencies.

## 6. Verify The Service Locally First

Run it directly once before creating the background service:

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

In another shell on the same machine:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

Wait for `/health/ready` to return:

```json
{
  "ok": true,
  "status": "ready",
  "model": "nvidia/parakeet-tdt-0.6b-v3",
  "detail": null
}
```

Stop the foreground process after the check.

## 7. Create A systemd Service

Create `/etc/systemd/system/transcription-service.service`:

```ini
[Unit]
Description=Pending App Transcription Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=transcribe
Group=transcribe
WorkingDirectory=/home/transcribe/transcription-service
Environment=PATH=/home/transcribe/.local/bin:/home/transcribe/transcription-service/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/transcribe/.local/bin/uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now transcription-service
```

Check logs:

```bash
sudo journalctl -u transcription-service -f
```

## 8. Expose It Securely With Caddy

Point your DNS record at the GPU host first.

Example DNS:

- `transcribe.example.com -> <your-server-public-ip>`

Then create `/etc/caddy/Caddyfile`:

```caddy
transcribe.example.com {
    encode gzip zstd

    reverse_proxy 127.0.0.1:8000
}
```

Validate and reload:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy will obtain and renew TLS certificates automatically.

At this point the service will be reachable at:

```text
https://transcribe.example.com
```

## 9. Lock Down The Firewall

Allow only SSH and HTTPS to the machine.

Example with UFW:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp
sudo ufw enable
```

Important:

- do not expose `8000` publicly
- only Caddy should be public
- keep the FastAPI app bound to `127.0.0.1`

## 10. Test The Public Endpoint

Health check:

```bash
curl https://transcribe.example.com/health/ready
```

Transcription request:

```bash
curl -X POST "https://transcribe.example.com/transcribe" \
  -H "Authorization: Bearer <TRANSCRIPTION_SERVICE_TOKEN>" \
  -F "audio=@/absolute/path/to/sample.wav" \
  -F "languageHint=auto" \
  -F "source=pending-app"
```

Expected success shape:

```json
{
  "ok": true,
  "transcript": "Hola, estoy probando el servicio.",
  "language": "es"
}
```

## 11. Point The Main App At The Deployed Service

On the app server, set:

```env
TRANSCRIPTION_SERVICE_URL=https://transcribe.example.com
TRANSCRIPTION_SERVICE_TOKEN=<same-random-token>
```

Restart the app server after updating env.

If you want to verify the whole broker path from the app repo:

```bash
TRANSCRIPTION_SMOKE_AUDIO_FILE=/absolute/path/to/sample.wav pnpm test:transcription-live
```

## 12. Updating The Service Later

From the service directory:

```bash
git pull
uv sync --extra inference
sudo systemctl restart transcription-service
```

Watch startup:

```bash
sudo journalctl -u transcription-service -f
```

## 13. Token Rotation

To rotate the token safely:

1. Generate a new random token.
2. Update the transcription service `.env`.
3. Update the app server `.env`.
4. Restart both services.
5. Re-run a health check and one live transcription smoke test.

## 14. Operational Notes

- the first startup may be slow because model weights may need to download
- `/health/ready` is the correct readiness signal, not just process start
- keep an eye on disk usage under the Hugging Face cache directory
- if GPU memory is tight, keep concurrency low as currently implemented
- bearer token auth is required, but HTTPS is still mandatory if the service is reachable over the public internet

## 15. Safer Alternative: Private Network Only

If you want it reachable anywhere without making it public, use a private overlay such as Tailscale and keep Caddy or the app itself bound only to the Tailscale interface.

That gives you:

- private network reachability from anywhere
- less exposure than a fully public endpoint
- the same bearer token auth on top

If you do that, the app server can target the Tailscale URL or private IP instead of a public DNS name.
