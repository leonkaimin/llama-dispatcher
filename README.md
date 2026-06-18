# local-llm-dispatcher

FastAPI dispatcher that sits between OpenClaw and one or more OpenAI-compatible `llama-server` instances.

It exposes:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

The dispatcher can route to a default local server, optional worker servers, and SSH-start a worker systemd service only when that worker is idle.

## Files

```text
dispatcher.py
dispatcher-config.example.json
requirements.txt
local-llm-dispatcher.service
local-llm-dispatcher.env.example
```

Do not commit your real `dispatcher-config.json`; it may contain private IPs, SSH users, and service names. It is ignored by `.gitignore`.

## Local Install

For a user-local checkout:

```bash
git clone https://github.com/YOUR_ORG/local-llm-dispatcher.git
cd local-llm-dispatcher
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp dispatcher-config.example.json dispatcher-config.json
```

Edit `dispatcher-config.json`, then run:

```bash
LLAMA_DISPATCHER_CONFIG="$PWD/dispatcher-config.json" \
  .venv/bin/uvicorn dispatcher:app --host 127.0.0.1 --port 8090
```

## systemd Install

This unit assumes the app is installed at `/opt/local-llm-dispatcher` and config is stored under `/etc/local-llm-dispatcher`.

```bash
sudo useradd --system --home /opt/local-llm-dispatcher --shell /usr/sbin/nologin local-llm-dispatcher
sudo mkdir -p /opt/local-llm-dispatcher /etc/local-llm-dispatcher
sudo cp -a . /opt/local-llm-dispatcher/
sudo chown -R local-llm-dispatcher:local-llm-dispatcher /opt/local-llm-dispatcher

cd /opt/local-llm-dispatcher
sudo -u local-llm-dispatcher python3 -m venv .venv
sudo -u local-llm-dispatcher .venv/bin/pip install -r requirements.txt

sudo cp dispatcher-config.example.json /etc/local-llm-dispatcher/dispatcher-config.json
sudo cp local-llm-dispatcher.env.example /etc/default/local-llm-dispatcher
sudo cp local-llm-dispatcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now local-llm-dispatcher.service
```

After editing config:

```bash
sudo systemctl restart local-llm-dispatcher.service
```

Logs:

```bash
journalctl -u local-llm-dispatcher.service -f
```

## Config

Copy the example config and edit it for your machines:

```bash
cp dispatcher-config.example.json dispatcher-config.json
```

Config shape:

```json
{
  "http": {
    "request_timeout_seconds": 1800,
    "connect_timeout_seconds": 10
  },
  "routing": {
    "short_prompt_char_limit": 500,
    "long_prompt_char_limit": 60000,
    "gpu_util_limit": 30,
    "gpu_mem_limit_mb": 6000,
    "idle_cache_seconds": 10,
    "ready_timeout_seconds": 45,
    "busy_process_pattern": "(ComfyUI|Wan|flux|sdxl|diffusers|torchrun|python.*main\\.py|python.*infer)",
    "auto_default_hint_pattern": "(long context|compact|full repo|codebase|whole repo|entire repo|many files|大量檔案|大量文件|整個 repo|整個專案|完整專案|壓縮上下文)"
  },
  "backends": [
    {
      "id": "main",
      "base": "http://127.0.0.1:8080/v1",
      "model": "main-model",
      "prefixes": ["main"],
      "default": true,
      "priority": 0
    },
    {
      "id": "worker",
      "base": "http://WORKER_HOST_OR_IP:8081/v1",
      "model": "worker-model",
      "prefixes": ["worker"],
      "ssh": "USER@WORKER_HOST_OR_IP",
      "service": "llama-worker.service",
      "idle_check": true,
      "auto_candidate": true,
      "priority": 10
    }
  ]
}
```

Backend fields:

```text
id               routing/log name
base             llama-server OpenAI-compatible /v1 base URL
model            model name sent to that llama-server
prefixes         request model prefixes that force this backend
default          fallback backend; set exactly one to true
ssh              SSH target for cold-start and idle checks
service          systemd service to start over SSH when not alive
idle_check       whether to check process list and nvidia-smi before use
auto_candidate   whether auto-local may route to this backend
priority         lower number is tried first for auto-local
```

## Routing

Default behavior:

```text
model starts with default backend prefix -> default backend
model starts with worker prefix          -> that worker if idle/ready, otherwise default
model is auto-local                      -> default for short/long/repo-wide prompts, otherwise first idle/ready auto candidate
anything else                            -> default backend
```

Worker idle check uses SSH:

```text
ps -eo args=
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
```

A worker is busy if a configured busy process is present, GPU utilization is over the limit, or GPU memory is over the limit.

## Test

Health:

```bash
curl -s http://127.0.0.1:8090/health | jq
```

Models:

```bash
curl -s http://127.0.0.1:8090/v1/models | jq
```

Non-streaming chat:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto-local",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "stream": false
  }' | jq
```

Streaming chat:

```bash
curl -N http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto-local",
    "messages": [{"role": "user", "content": "Write a short haiku about local inference."}],
    "stream": true
  }'
```

## OpenClaw Provider Config

```json
{
  "providers": {
    "local-dispatcher": {
      "baseUrl": "http://127.0.0.1:8090/v1",
      "apiKey": "dummy",
      "api": "openai-chat-completions",
      "timeoutSeconds": 1800
    }
  }
}
```

## Worker sudoers

On each worker host, allow only the exact llama service commands needed by the SSH user.

Create a sudoers drop-in:

```bash
sudo visudo -f /etc/sudoers.d/llama-worker
```

Example:

```text
USER ALL=(root) NOPASSWD: /bin/systemctl start llama-worker.service, /bin/systemctl stop llama-worker.service, /bin/systemctl status llama-worker.service
USER ALL=(root) NOPASSWD: /usr/bin/systemctl start llama-worker.service, /usr/bin/systemctl stop llama-worker.service, /usr/bin/systemctl status llama-worker.service
```

Test from the dispatcher host:

```bash
ssh USER@WORKER_HOST_OR_IP 'sudo systemctl status llama-worker.service'
ssh USER@WORKER_HOST_OR_IP 'sudo systemctl start llama-worker.service'
```
