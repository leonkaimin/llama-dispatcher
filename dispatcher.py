import asyncio
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("local-llm-dispatcher")
app = FastAPI(title="local-llm-dispatcher")


CONFIG_PATH = os.getenv(
    "LLAMA_DISPATCHER_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispatcher-config.json"),
)


def load_config_file() -> dict[str, Any]:
    if not CONFIG_PATH or not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("dispatcher config must be a JSON object")
    return data


try:
    CONFIG = load_config_file()
except Exception as exc:
    logger.error(json.dumps({"error": "invalid_config_file", "path": CONFIG_PATH, "detail": str(exc)}))
    CONFIG = {}


def config_get(path: tuple[str, ...], default: Any) -> Any:
    value: Any = CONFIG
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def setting_str(env_name: str, path: tuple[str, ...], default: str) -> str:
    return str(os.getenv(env_name, config_get(path, default)))


def setting_int(env_name: str, path: tuple[str, ...], default: int) -> int:
    return int(os.getenv(env_name, config_get(path, default)))


def setting_float(env_name: str, path: tuple[str, ...], default: float) -> float:
    return float(os.getenv(env_name, config_get(path, default)))


GB10_BASE = setting_str("GB10_BASE", ("legacy_defaults", "gb10_base"), "http://127.0.0.1:8080/v1").rstrip("/")
PC_BASE = setting_str("PC_BASE", ("legacy_defaults", "pc_base"), "http://127.0.0.1:8081/v1").rstrip("/")
PC_SSH = setting_str("PC_SSH", ("legacy_defaults", "pc_ssh"), "")
PC_SERVICE = setting_str("PC_SERVICE", ("legacy_defaults", "pc_service"), "llama-worker.service")
GB10_MODEL = setting_str("GB10_MODEL", ("legacy_defaults", "gb10_model"), "main-model")
PC_MODEL = setting_str("PC_MODEL", ("legacy_defaults", "pc_model"), "worker-model")
PC_GPU_UTIL_LIMIT = setting_int("PC_GPU_UTIL_LIMIT", ("routing", "gpu_util_limit"), 30)
PC_GPU_MEM_LIMIT_MB = setting_int("PC_GPU_MEM_LIMIT_MB", ("routing", "gpu_mem_limit_mb"), 6000)
PC_IDLE_CACHE_SECONDS = setting_float("PC_IDLE_CACHE_SECONDS", ("routing", "idle_cache_seconds"), 10)
PC_READY_TIMEOUT_SECONDS = setting_float("PC_READY_TIMEOUT_SECONDS", ("routing", "ready_timeout_seconds"), 45)
SHORT_PROMPT_CHAR_LIMIT = setting_int("SHORT_PROMPT_CHAR_LIMIT", ("routing", "short_prompt_char_limit"), 500)
LONG_PROMPT_CHAR_LIMIT = setting_int("LONG_PROMPT_CHAR_LIMIT", ("routing", "long_prompt_char_limit"), 60000)

REQUEST_TIMEOUT_SECONDS = setting_float("REQUEST_TIMEOUT_SECONDS", ("http", "request_timeout_seconds"), 1800)
CONNECT_TIMEOUT_SECONDS = setting_float("CONNECT_TIMEOUT_SECONDS", ("http", "connect_timeout_seconds"), 10)

BUSY_PROCESS_PATTERN = setting_str(
    "BUSY_PROCESS_PATTERN",
    ("routing", "busy_process_pattern"),
    r"(ComfyUI|Wan|flux|sdxl|diffusers|torchrun|python.*main\.py|python.*infer)",
)
AUTO_GB10_HINT_PATTERN = setting_str(
    "AUTO_GB10_HINT_PATTERN",
    ("routing", "auto_default_hint_pattern"),
    r"(long context|compact|full repo|codebase|whole repo|entire repo|"
    r"many files|大量檔案|大量文件|整個 repo|整個專案|完整專案|壓縮上下文)",
)
BUSY_PROCESS_RE = re.compile(BUSY_PROCESS_PATTERN, re.IGNORECASE)
AUTO_GB10_HINT_RE = re.compile(AUTO_GB10_HINT_PATTERN, re.IGNORECASE)


@dataclass(frozen=True)
class Backend:
    id: str
    base: str
    model: str
    prefixes: tuple[str, ...] = field(default_factory=tuple)
    ssh: str | None = None
    service: str | None = None
    idle_check: bool = False
    auto_candidate: bool = False
    stop_after_request: bool = False
    default: bool = False
    priority: int = 100
    gpu_util_limit: int = PC_GPU_UTIL_LIMIT
    gpu_mem_limit_mb: int = PC_GPU_MEM_LIMIT_MB
    idle_cache_seconds: float = PC_IDLE_CACHE_SECONDS
    ready_timeout_seconds: float = PC_READY_TIMEOUT_SECONDS
    stop_timeout_seconds: float = 15


@dataclass
class IdleCache:
    expires_at: float = 0
    value: bool = False
    reason: str = "not_checked"


idle_caches: dict[str, IdleCache] = {}
idle_lock = asyncio.Lock()


def parse_backend(item: dict[str, Any]) -> Backend:
    prefixes = item.get("prefixes", item.get("model_prefixes", []))
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    backend_id = str(item["id"])
    return Backend(
        id=backend_id,
        base=str(item["base"]).rstrip("/"),
        model=str(item["model"]),
        prefixes=tuple(str(prefix) for prefix in prefixes),
        ssh=item.get("ssh"),
        service=item.get("service"),
        idle_check=bool(item.get("idle_check", False)),
        auto_candidate=bool(item.get("auto_candidate", False)),
        stop_after_request=bool(item.get("stop_after_request", False)),
        default=bool(item.get("default", False)),
        priority=int(item.get("priority", 100)),
        gpu_util_limit=int(item.get("gpu_util_limit", PC_GPU_UTIL_LIMIT)),
        gpu_mem_limit_mb=int(item.get("gpu_mem_limit_mb", PC_GPU_MEM_LIMIT_MB)),
        idle_cache_seconds=float(item.get("idle_cache_seconds", PC_IDLE_CACHE_SECONDS)),
        ready_timeout_seconds=float(item.get("ready_timeout_seconds", PC_READY_TIMEOUT_SECONDS)),
        stop_timeout_seconds=float(item.get("stop_timeout_seconds", 15)),
    )


def default_backends() -> list[Backend]:
    return [
        Backend(
            id="main",
            base=GB10_BASE,
            model=GB10_MODEL,
            prefixes=("main",),
            default=True,
            priority=0,
        ),
        Backend(
            id="worker",
            base=PC_BASE,
            model=PC_MODEL,
            prefixes=("worker",),
            ssh=PC_SSH,
            service=PC_SERVICE,
            idle_check=True,
            auto_candidate=True,
            stop_after_request=True,
            priority=10,
        ),
    ]


def load_backends() -> list[Backend]:
    raw = os.getenv("LLAMA_BACKENDS_JSON", "").strip()
    try:
        if raw:
            data = json.loads(raw)
            source = "LLAMA_BACKENDS_JSON"
        else:
            data = CONFIG.get("backends")
            source = CONFIG_PATH
        if data is None:
            return default_backends()
        if not isinstance(data, list):
            raise ValueError(f"backend config from {source} must be a JSON list")
        backends = [parse_backend(item) for item in data]
    except Exception as exc:
        logger.error(json.dumps({"error": "invalid_backends_config", "detail": str(exc)}))
        return default_backends()
    if not backends:
        return default_backends()
    if not any(backend.default for backend in backends):
        first = backends[0]
        backends[0] = Backend(**{**first.__dict__, "default": True})
    return backends


BACKENDS = load_backends()
BACKENDS_BY_ID = {backend.id: backend for backend in BACKENDS}
DEFAULT_BACKEND = next((backend for backend in BACKENDS if backend.default), BACKENDS[0])
AUTO_CANDIDATES = sorted(
    [backend for backend in BACKENDS if backend.auto_candidate and not backend.default],
    key=lambda backend: backend.priority,
)


def model_card(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "local-llm-dispatcher",
    }


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(extract_text(item.get("text") or item.get("content")))
            else:
                parts.append(extract_text(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return "\n".join(extract_text(v) for v in value.values())
    return str(value)


def prompt_text(payload: dict[str, Any]) -> str:
    parts = []
    for message in payload.get("messages", []):
        if isinstance(message, dict):
            parts.append(extract_text(message.get("content")))
        else:
            parts.append(extract_text(message))
    return "\n".join(part for part in parts if part)


async def run_ssh(backend: Backend, command: str, timeout: float = 8) -> tuple[int, str, str]:
    if not backend.ssh:
        return 1, "", "backend has no ssh target"
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        backend.ssh,
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", "ssh timeout"
    return (
        proc.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def check_backend_idle_uncached(backend: Backend) -> tuple[bool, str]:
    if not backend.idle_check:
        return True, "idle_check_disabled"
    command = (
        "sh -lc '"
        "ps -eo args=; "
        "echo __GPU__; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; "
        "else echo NO_NVIDIA_SMI; fi"
        "'"
    )
    code, stdout, stderr = await run_ssh(backend, command)
    if code != 0:
        return False, f"ssh_failed:{stderr.strip() or code}"

    process_part, _, gpu_part = stdout.partition("__GPU__")
    for line in process_part.splitlines():
        if BUSY_PROCESS_RE.search(line):
            return False, "busy_process"

    if "NO_NVIDIA_SMI" in gpu_part:
        return False, "nvidia_smi_missing"

    for line in gpu_part.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            gpu_util = int(fields[0])
            mem_used = int(fields[1])
        except ValueError:
            continue
        if gpu_util > backend.gpu_util_limit:
            return False, f"gpu_util_gt_{backend.gpu_util_limit}"
        if mem_used > backend.gpu_mem_limit_mb:
            return False, f"gpu_mem_gt_{backend.gpu_mem_limit_mb}"

    return True, "idle"


async def check_backend_idle(backend: Backend) -> tuple[bool, str]:
    now = time.monotonic()
    async with idle_lock:
        cache = idle_caches.setdefault(backend.id, IdleCache())
        if now < cache.expires_at:
            return cache.value, f"cached_{cache.reason}"
        value, reason = await check_backend_idle_uncached(backend)
        cache.value = value
        cache.reason = reason
        cache.expires_at = now + backend.idle_cache_seconds
        return value, reason


async def endpoint_alive(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=3)) as client:
            response = await client.get(f"{base_url}/models")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def ensure_backend_ready(backend: Backend) -> tuple[bool, str]:
    if await endpoint_alive(backend.base):
        return True, f"{backend.id}_already_alive"
    if not backend.service:
        return False, f"{backend.id}_not_alive"

    service = shlex.quote(backend.service)
    code, _, stderr = await run_ssh(
        backend,
        f"sudo systemctl start {service}",
        timeout=15,
    )
    if code != 0:
        return False, f"{backend.id}_start_failed:{stderr.strip() or code}"

    deadline = time.monotonic() + backend.ready_timeout_seconds
    while time.monotonic() < deadline:
        if await endpoint_alive(backend.base):
            return True, f"{backend.id}_started_ready"
        await asyncio.sleep(1)
    return False, f"{backend.id}_ready_timeout"


async def stop_backend_after_request(backend: Backend, reason: str) -> None:
    if not backend.stop_after_request or not backend.service or not backend.ssh:
        return

    service = shlex.quote(backend.service)
    code, _, stderr = await run_ssh(
        backend,
        f"sudo systemctl stop {service}",
        timeout=backend.stop_timeout_seconds,
    )
    logger.info(
        json.dumps(
            {
                "route": backend.id,
                "action": "stop_service",
                "service": backend.service,
                "ok": code == 0,
                "reason": reason,
                "error": stderr.strip() or None,
            },
            ensure_ascii=False,
        )
    )


def backend_for_requested_model(requested_model: str) -> Backend | None:
    matches: list[tuple[int, Backend]] = []
    for backend in BACKENDS:
        for prefix in backend.prefixes:
            if requested_model.startswith(prefix):
                matches.append((len(prefix), backend))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


async def prepare_backend(backend: Backend) -> tuple[bool, bool | None, str]:
    idle, idle_reason = await check_backend_idle(backend)
    if not idle:
        return False, idle, idle_reason
    ready, ready_reason = await ensure_backend_ready(backend)
    if not ready:
        return False, idle, ready_reason
    return True, idle, ready_reason


async def choose_route(payload: dict[str, Any]) -> tuple[Backend, bool | None, str]:
    requested_model = str(payload.get("model") or "")
    text = prompt_text(payload)
    text_len = len(text)

    forced_backend = backend_for_requested_model(requested_model)
    if forced_backend:
        if forced_backend.default:
            return forced_backend, None, f"model_{forced_backend.id}_prefix"
        ready, idle, reason = await prepare_backend(forced_backend)
        if ready:
            return forced_backend, idle, reason
        return DEFAULT_BACKEND, idle, reason

    if requested_model == "auto-local":
        if text_len > LONG_PROMPT_CHAR_LIMIT:
            return DEFAULT_BACKEND, None, "auto_prompt_too_long"
        if text_len < SHORT_PROMPT_CHAR_LIMIT:
            return DEFAULT_BACKEND, None, "auto_prompt_too_short"
        if AUTO_GB10_HINT_RE.search(text):
            return DEFAULT_BACKEND, None, "auto_long_context_hint"

        for backend in AUTO_CANDIDATES:
            ready, idle, reason = await prepare_backend(backend)
            if ready:
                return backend, idle, f"auto_{backend.id}_idle"
            logger.info(
                json.dumps(
                    {
                        "route": DEFAULT_BACKEND.id,
                        "candidate": backend.id,
                        "pc_idle": idle,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
            )
        return DEFAULT_BACKEND, None, "auto_no_candidate_ready"

    return DEFAULT_BACKEND, None, "default"


def route_log(route: Backend, stream: bool, pc_idle: bool | None, reason: str) -> None:
    logger.info(
        json.dumps(
            {
                "route": route.id,
                "stream": stream,
                "model": route.model,
                "pc_idle": pc_idle,
                "reason": reason,
            },
            ensure_ascii=False,
        )
    )


def headers_for_upstream(request: Request) -> dict[str, str]:
    headers = {}
    for key, value in request.headers.items():
        lowered = key.lower()
        if lowered in {"host", "content-length", "connection"}:
            continue
        headers[key] = value
    return headers


async def post_non_streaming(
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)


async def stream_upstream(
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    fail_on_5xx: bool = False,
):
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if fail_on_5xx and response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    "upstream stream failed",
                    request=response.request,
                    response=response,
                )
            async for chunk in response.aiter_bytes():
                yield chunk


async def stream_with_optional_fallback(
    route: Backend,
    primary_payload: dict[str, Any],
    headers: dict[str, str],
    pc_idle: bool | None,
):
    yielded = False
    stopped = False
    try:
        async for chunk in stream_upstream(
            route.base,
            primary_payload,
            headers,
            fail_on_5xx=(route != DEFAULT_BACKEND),
        ):
            yielded = True
            yield chunk
    except httpx.HTTPError as exc:
        if route != DEFAULT_BACKEND and not yielded:
            await stop_backend_after_request(route, "stream_failed_before_fallback")
            stopped = True
            fallback_payload = dict(primary_payload)
            fallback_payload["model"] = DEFAULT_BACKEND.model
            route_log(DEFAULT_BACKEND, True, pc_idle, f"{route.id}_stream_failed_fallback")
            async for chunk in stream_upstream(DEFAULT_BACKEND.base, fallback_payload, headers):
                yield chunk
            return
        error = {"error": {"message": str(exc), "type": "upstream_stream_error"}}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n".encode()
    finally:
        if not stopped:
            await stop_backend_after_request(route, "stream_complete")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "local-llm-dispatcher",
        "default_backend": DEFAULT_BACKEND.id,
        "backends": [
            {
                "id": backend.id,
                "base": backend.base,
                "model": backend.model,
                "default": backend.default,
                "auto_candidate": backend.auto_candidate,
                "idle_check": backend.idle_check,
                "stop_after_request": backend.stop_after_request,
            }
            for backend in BACKENDS
        ],
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    seen = set()
    model_ids = []
    for backend in BACKENDS:
        if backend.model not in seen:
            seen.add(backend.model)
            model_ids.append(backend.model)
    model_ids.append("auto-local")
    return {
        "object": "list",
        "data": [model_card(model_id) for model_id in model_ids],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        original_payload = await request.json()
    except Exception as exc:
        return JSONResponse({"error": f"invalid JSON body: {exc}"}, status_code=400)
    if not isinstance(original_payload, dict):
        return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)

    stream = bool(original_payload.get("stream"))
    route, pc_idle, reason = await choose_route(original_payload)
    headers = headers_for_upstream(request)

    selected_payload = dict(original_payload)
    selected_payload["model"] = route.model
    route_log(route, stream, pc_idle, reason)
    stopped = False

    if stream:
        return StreamingResponse(
            stream_with_optional_fallback(route, selected_payload, headers, pc_idle),
            media_type="text/event-stream",
        )

    try:
        response = await post_non_streaming(route.base, selected_payload, headers)
        if route != DEFAULT_BACKEND and response.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"{route.id} upstream failed",
                request=response.request,
                response=response,
            )
    except httpx.HTTPError as exc:
        if route == DEFAULT_BACKEND:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "default_upstream_error"}},
                status_code=502,
            )
        await stop_backend_after_request(route, "request_failed_before_fallback")
        stopped = True
        selected_payload = dict(original_payload)
        selected_payload["model"] = DEFAULT_BACKEND.model
        route_log(DEFAULT_BACKEND, stream, pc_idle, f"{route.id}_request_failed_fallback")
        try:
            response = await post_non_streaming(DEFAULT_BACKEND.base, selected_payload, headers)
        except httpx.HTTPError as default_exc:
            return JSONResponse(
                {"error": {"message": str(default_exc), "type": "default_upstream_error"}},
                status_code=502,
            )

    if not stopped:
        await stop_backend_after_request(route, "request_complete")

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )
