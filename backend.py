"""Selected-region image classifier backed by an OpenAI-compatible VLM server.

The service can launch llama-server for CPU inference, launch vLLM for CUDA, or
connect to an already-running endpoint through VLM_URL. Run one Uvicorn worker
when this process owns the inference-engine child process.
"""

import asyncio
import base64
import json
import logging
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import uuid
import warnings
from asyncio.subprocess import Process
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import monotonic, perf_counter
from typing import Annotated, Any, Literal, TypeVar

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageOps, ImageStat

LOGGER = logging.getLogger("lfm25-vl")
ProgressCallback = Callable[[str, str, int, dict[str, Any] | None], Awaitable[None]]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_list(name: str, default: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, default).split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    model_native: str = os.getenv("MODEL", "LiquidAI/LFM2.5-VL-450M")
    model_gguf: str = os.getenv("MODEL_GGUF", "LiquidAI/LFM2.5-VL-450M-GGUF:Q4_K_M")
    api_model: str = os.getenv("API_MODEL", os.getenv("MODEL", "LiquidAI/LFM2.5-VL-450M"))
    device: str = os.getenv("DEVICE", "cpu").lower()
    engine: str = os.getenv("ENGINE", "auto").lower()
    vlm_url: str = os.getenv("VLM_URL", "").rstrip("/")
    engine_host: str = os.getenv("ENGINE_HOST", "127.0.0.1")
    engine_port: int = int(os.getenv("ENGINE_PORT", "8080"))
    threads: int = int(os.getenv("THREADS", "4"))
    concurrency: int = int(os.getenv("CONCURRENCY", "2"))
    preprocess_concurrency: int = int(os.getenv("PREPROCESS_CONCURRENCY", "2"))
    context_tokens_per_slot: int = int(os.getenv("CTX_PER_SLOT", "256"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "256"))
    ubatch_size: int = int(os.getenv("UBATCH_SIZE", "256"))
    image_tokens: int = int(os.getenv("IMAGE_TOKENS", "64"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "8"))
    crop_padding: float = float(os.getenv("CROP_PADDING", "0.08"))
    image_side: int = int(os.getenv("IMAGE_SIDE", "384"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "88"))
    max_upload_bytes: int = int(os.getenv("MAX_BYTES", str(100 * 1024 * 1024)))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", str(40_000_000)))
    image_session_ttl: int = int(os.getenv("IMAGE_SESSION_TTL", "900"))
    image_session_cache_bytes: int = int(
        os.getenv("IMAGE_SESSION_CACHE_BYTES", str(256 * 1024 * 1024))
    )
    max_image_sessions: int = int(os.getenv("MAX_IMAGE_SESSIONS", "32"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "300"))
    startup_timeout: float = float(os.getenv("START_TIMEOUT", "600"))
    resample: str = os.getenv("RESAMPLE", "bilinear").lower()
    logprobs: bool = _env_bool("LOGPROBS", True)
    gpu_layers: str = os.getenv("GPU_LAYERS", "all")
    dtype: str = os.getenv("DTYPE", "bfloat16").lower()
    llama_server: str = os.getenv("LLAMA_SERVER", "llama-server")
    llama_extra_args: tuple[str, ...] = tuple(shlex.split(os.getenv("LLAMA_SERVER_EXTRA_ARGS", "")))
    vllm_binary: str = os.getenv("VLLM", "vllm")
    cors_origins: tuple[str, ...] = _env_list("CORS_ORIGINS", "")
    prompt: str = os.getenv(
        "PROMPT",
        "Name the single main visible subject in this image. "
        "The subject may be a person, animal, body part, or object. "
        "Reply with only the most specific common English noun phrase, "
        "lowercase, one to four words. If a whole human is the main subject, "
        "reply person. No article, punctuation, explanation, brand, scene, "
        "material, or unsupported adjective. If unclear, reply unknown.",
    )

    def validate(self) -> None:
        positive = {
            "THREADS": self.threads,
            "CONCURRENCY": self.concurrency,
            "PREPROCESS_CONCURRENCY": self.preprocess_concurrency,
            "CTX_PER_SLOT": self.context_tokens_per_slot,
            "IMAGE_SIDE": self.image_side,
            "MAX_BYTES": self.max_upload_bytes,
            "MAX_IMAGE_PIXELS": self.max_image_pixels,
            "IMAGE_SESSION_TTL": self.image_session_ttl,
            "IMAGE_SESSION_CACHE_BYTES": self.image_session_cache_bytes,
            "MAX_IMAGE_SESSIONS": self.max_image_sessions,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise RuntimeError(f"These settings must be positive: {', '.join(invalid)}")
        if self.resample not in RESAMPLING_FILTERS:
            supported = ", ".join(sorted(RESAMPLING_FILTERS))
            raise RuntimeError(f"RESAMPLE must be one of: {supported}")
        if not 1 <= self.jpeg_quality <= 95:
            raise RuntimeError("JPEG_QUALITY must be between 1 and 95")


RESAMPLING_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
SETTINGS = Settings()
SETTINGS.validate()


@dataclass(frozen=True, slots=True)
class Region:
    """Normalized rectangle or circle selected by the user."""

    shape: Literal["rect", "circle"]
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    def validate(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in coordinates):
            raise HTTPException(400, "Coordinates must be finite and between 0 and 1")
        if self.width < 0.01 or self.height < 0.01:
            raise HTTPException(400, "Selected region is too small")

    def pixel_box(
        self, image_size: tuple[int, int], padding: float = 0
    ) -> tuple[int, int, int, int]:
        image_width, image_height = image_size
        left, top, right, bottom = self.box
        width = (right - left) * image_width
        height = (bottom - top) * image_height
        margin = padding * max(width, height)
        return (
            max(0, math.floor(left * image_width - margin)),
            max(0, math.floor(top * image_height - margin)),
            min(image_width, math.ceil(right * image_width + margin)),
            min(image_height, math.ceil(bottom * image_height + margin)),
        )


@dataclass(frozen=True, slots=True)
class PreparedImage:
    image: Image.Image
    details: dict[str, Any]


@dataclass(slots=True)
class ImageSession:
    session_id: str
    data: bytes
    width: int
    height: int
    created_at: float
    expires_at: float

    @property
    def upload_bytes(self) -> int:
        return len(self.data)


class ImageSessionStore:
    """Bounded process-local storage for immutable uploaded image bytes."""

    def __init__(
        self,
        max_bytes: int,
        max_sessions: int,
        ttl_seconds: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, ImageSession] = OrderedDict()
        self._total_bytes = 0
        self._lock = asyncio.Lock()

    def _remove_unlocked(self, session_id: str) -> bool:
        session = self._items.pop(session_id, None)
        if session is None:
            return False
        self._total_bytes -= session.upload_bytes
        return True

    def _purge_expired_unlocked(self, now: float) -> None:
        expired = [
            session_id for session_id, session in self._items.items() if session.expires_at <= now
        ]
        for session_id in expired:
            self._remove_unlocked(session_id)

    async def put(
        self,
        data: bytes,
        width: int,
        height: int,
    ) -> ImageSession:
        if len(data) > self.max_bytes:
            raise HTTPException(413, "Image is larger than the session cache capacity")

        async with self._lock:
            now = self._clock()
            self._purge_expired_unlocked(now)
            while self._items and (
                len(self._items) >= self.max_sessions
                or self._total_bytes + len(data) > self.max_bytes
            ):
                oldest_id = next(iter(self._items))
                self._remove_unlocked(oldest_id)

            session = ImageSession(
                session_id=secrets.token_urlsafe(32),
                data=data,
                width=width,
                height=height,
                created_at=now,
                expires_at=now + self.ttl_seconds,
            )
            self._items[session.session_id] = session
            self._total_bytes += session.upload_bytes
            return session

    async def get(self, session_id: str) -> ImageSession:
        async with self._lock:
            now = self._clock()
            self._purge_expired_unlocked(now)
            session = self._items.get(session_id)
            if session is None:
                raise HTTPException(404, "Image session is missing or expired")
            session.expires_at = now + self.ttl_seconds
            self._items.move_to_end(session_id)
            return session

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._remove_unlocked(session_id)

    async def cleanup(self) -> None:
        async with self._lock:
            self._purge_expired_unlocked(self._clock())

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()
            self._total_bytes = 0

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            self._purge_expired_unlocked(self._clock())
            return {
                "sessions": len(self._items),
                "bytes": self._total_bytes,
                "max_sessions": self.max_sessions,
                "max_bytes": self.max_bytes,
                "ttl_seconds": self.ttl_seconds,
            }


@dataclass(slots=True)
class EngineState:
    engine: str
    device: str
    base_url: str
    api_model: str
    command: list[str]
    process: Process | None = None
    client: httpx.AsyncClient | None = None
    startup: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    binary_version: str | None = None


class InferenceGate:
    """Semaphore with queue counters suitable for the browser status display."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._semaphore = asyncio.Semaphore(self.limit)
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0

    def _snapshot_unlocked(self) -> dict[str, int]:
        return {
            "limit": self.limit,
            "waiting": self._waiting,
            "active": self._active,
            "completed": self._completed,
            "failed": self._failed,
            "cancelled": self._cancelled,
        }

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return self._snapshot_unlocked()

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[dict[str, int], None]:
        acquired = False
        async with self._lock:
            self._waiting += 1
            queued_ahead = max(0, self._waiting - 1)

        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._waiting -= 1
                self._active += 1
                running = self._snapshot_unlocked()
            yield {"queued_ahead": queued_ahead, **running}
        except asyncio.CancelledError:
            async with self._lock:
                self._cancelled += 1
            raise
        except Exception:
            async with self._lock:
                self._failed += 1
            raise
        else:
            async with self._lock:
                self._completed += 1
        finally:
            async with self._lock:
                if acquired:
                    self._active = max(0, self._active - 1)
                else:
                    self._waiting = max(0, self._waiting - 1)
            if acquired:
                self._semaphore.release()


def parse_region(
    x1: Annotated[float, Form()],
    y1: Annotated[float, Form()],
    x2: Annotated[float, Form()],
    y2: Annotated[float, Form()],
    shape: Annotated[Literal["rect", "circle"], Form()] = "rect",
) -> Region:
    region = Region(shape, x1, y1, x2, y2)
    region.validate()
    return region


def resolve_device() -> str:
    if SETTINGS.device == "auto":
        return "cuda" if shutil.which("nvidia-smi") else "cpu"
    if SETTINGS.device in {"gpu", "0"} or SETTINGS.device.startswith("cuda"):
        return "cuda"
    return "cpu"


def resolve_engine(device: str) -> str:
    if SETTINGS.vlm_url:
        return "external"
    if SETTINGS.engine == "auto":
        return "llama" if device == "cpu" else "vllm"
    if SETTINGS.engine not in {"llama", "vllm"}:
        raise RuntimeError("ENGINE must be auto, llama, or vllm")
    return SETTINGS.engine


def endpoint_url() -> str:
    return SETTINGS.vlm_url or f"http://{SETTINGS.engine_host}:{SETTINGS.engine_port}/v1"


def llama_command(device: str) -> list[str]:
    command = [
        SETTINGS.llama_server,
        "-hf",
        SETTINGS.model_gguf,
        "--alias",
        SETTINGS.api_model,
        "--host",
        SETTINGS.engine_host,
        "--port",
        str(SETTINGS.engine_port),
        "--threads",
        str(SETTINGS.threads),
        "--threads-batch",
        str(SETTINGS.threads),
        "--threads-http",
        str(max(2, SETTINGS.concurrency)),
        "--parallel",
        str(SETTINGS.concurrency),
        "--ctx-size",
        str(SETTINGS.context_tokens_per_slot * SETTINGS.concurrency),
        "--batch-size",
        str(SETTINGS.batch_size),
        "--ubatch-size",
        str(SETTINGS.ubatch_size),
        "--mtmd-batch-max-tokens",
        str(SETTINGS.image_tokens * SETTINGS.concurrency),
        "--cont-batching",
        "--cache-prompt",
        "--image-min-tokens",
        str(min(32, SETTINGS.image_tokens)),
        "--image-max-tokens",
        str(SETTINGS.image_tokens),
        "--gpu-layers",
        "0" if device == "cpu" else SETTINGS.gpu_layers,
    ]
    if device == "cpu":
        command.append("--no-mmproj-offload")
    command.extend(SETTINGS.llama_extra_args)
    return command


def vllm_command() -> list[str]:
    if SETTINGS.dtype not in {"bfloat16", "float16", "half"}:
        raise RuntimeError("DTYPE must be bfloat16 or float16")
    return [
        SETTINGS.vllm_binary,
        "serve",
        SETTINGS.model_native,
        "--host",
        SETTINGS.engine_host,
        "--port",
        str(SETTINGS.engine_port),
        "--dtype",
        SETTINGS.dtype,
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        str(SETTINGS.concurrency),
        "--limit-mm-per-prompt",
        '{"image":1}',
        "--mm-processor-kwargs",
        f'{{"max_image_tokens":{SETTINGS.image_tokens},"min_tiles":1,"max_tiles":1}}',
    ]


def engine_root(base_url: str) -> str:
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def executable_version(binary: str) -> str | None:
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output[:2000] or None


async def wait_until_ready(state: EngineState) -> dict[str, Any]:
    if state.client is None:
        raise RuntimeError("Inference HTTP client is not initialized")

    started = perf_counter()
    probe_urls = [f"{state.base_url}/models"]
    if state.engine != "external":
        probe_urls.insert(0, f"{engine_root(state.base_url)}/health")

    while perf_counter() - started < SETTINGS.startup_timeout:
        if state.process is not None and state.process.returncode is not None:
            raise RuntimeError(f"{state.engine} exited with code {state.process.returncode}")
        for url in probe_urls:
            try:
                response = await state.client.get(url, timeout=2)
                if response.status_code < 500:
                    return await fetch_engine_metadata(state)
            except httpx.HTTPError:
                continue
        await asyncio.sleep(0.5)

    raise RuntimeError(
        f"Inference engine was not ready within {SETTINGS.startup_timeout:g} seconds"
    )


async def fetch_engine_metadata(state: EngineState) -> dict[str, Any]:
    if state.client is None:
        return {}
    endpoints = {
        "models": f"{state.base_url}/models",
        "props": f"{engine_root(state.base_url)}/props",
    }
    metadata: dict[str, Any] = {}
    for name, url in endpoints.items():
        try:
            response = await state.client.get(url, timeout=5)
            response.raise_for_status()
            value = response.json()
            metadata[name] = value if isinstance(value, (dict, list)) else None
        except (httpx.HTTPError, ValueError):
            metadata[name] = None
    return metadata


async def start_engine(state: EngineState) -> None:
    started = perf_counter()
    state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(SETTINGS.request_timeout, connect=10),
        limits=httpx.Limits(
            max_connections=max(2, SETTINGS.concurrency),
            max_keepalive_connections=max(2, SETTINGS.concurrency),
        ),
    )

    if state.command:
        binary = state.command[0]
        if shutil.which(binary) is None:
            raise RuntimeError(f"Required executable not found: {binary}")
        state.binary_version = await asyncio.to_thread(executable_version, binary)
        LOGGER.info("Starting inference engine: %s", shlex.join(state.command))
        state.process = await asyncio.create_subprocess_exec(*state.command)

    state.metadata = await wait_until_ready(state)
    state.startup = {
        "engine_ready_ms": round((perf_counter() - started) * 1000, 3),
        "launched_by_backend": bool(state.command),
    }


async def stop_engine(state: EngineState) -> None:
    if state.client is not None:
        await state.client.aclose()
    if state.process is None or state.process.returncode is not None:
        return
    state.process.terminate()
    try:
        await asyncio.wait_for(state.process.wait(), timeout=10)
    except TimeoutError:
        state.process.kill()
        await state.process.wait()


@lru_cache(maxsize=1)
def lscpu_info() -> dict[str, Any]:
    """Return a concise, structured subset of lscpu --json."""
    if shutil.which("lscpu") is None:
        return {}
    try:
        result = subprocess.run(
            ["lscpu", "--json"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}

    fields = {
        str(item.get("field", "")).rstrip(":"): item.get("data")
        for item in payload.get("lscpu", [])
        if item.get("field")
    }
    flags = set(str(fields.get("Flags", "")).split())
    noteworthy_flags = [
        flag
        for flag in (
            "sse4_2",
            "avx",
            "avx2",
            "f16c",
            "fma",
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx_vnni",
        )
        if flag in flags
    ]
    return {
        "architecture": fields.get("Architecture"),
        "model_name": fields.get("Model name"),
        "vendor": fields.get("Vendor ID"),
        "logical_cpus": fields.get("CPU(s)"),
        "threads_per_core": fields.get("Thread(s) per core"),
        "cores_per_socket": fields.get("Core(s) per socket"),
        "sockets": fields.get("Socket(s)"),
        "numa_nodes": fields.get("NUMA node(s)"),
        "virtualization": fields.get("Virtualization"),
        "hypervisor": fields.get("Hypervisor vendor"),
        "max_mhz": fields.get("CPU max MHz"),
        "l1d_cache": fields.get("L1d cache"),
        "l1i_cache": fields.get("L1i cache"),
        "l2_cache": fields.get("L2 cache"),
        "l3_cache": fields.get("L3 cache"),
        "noteworthy_flags": noteworthy_flags,
        "flags_count": len(flags),
    }


def _proc_values(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _kib_to_bytes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"(\d+)", value)
    return int(match.group(1)) * 1024 if match else None


def memory_info() -> dict[str, int | None]:
    values = _proc_values("/proc/meminfo")
    total = _kib_to_bytes(values.get("MemTotal"))
    available = _kib_to_bytes(values.get("MemAvailable"))
    swap_total = _kib_to_bytes(values.get("SwapTotal"))
    swap_free = _kib_to_bytes(values.get("SwapFree"))
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": total - available if total and available is not None else None,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": (
            swap_total - swap_free if swap_total is not None and swap_free is not None else None
        ),
    }


def process_info(pid: int | None) -> dict[str, Any] | None:
    if pid is None:
        return None
    values = _proc_values(f"/proc/{pid}/status")
    return {
        "pid": pid,
        "rss_bytes": _kib_to_bytes(values.get("VmRSS")),
        "peak_rss_bytes": _kib_to_bytes(values.get("VmHWM")),
        "threads": int(values["Threads"]) if values.get("Threads", "").isdigit() else None,
    }


def cuda_info() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        first = result.stdout.strip().splitlines()[0]
        name, driver, memory_total, memory_used = [part.strip() for part in first.split(",")]
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return None
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(memory_total),
        "memory_used_mib": int(memory_used),
    }


def hardware_info(state: EngineState) -> dict[str, Any]:
    try:
        load_average = [round(value, 3) for value in os.getloadavg()]
    except (AttributeError, OSError):
        load_average = None
    disk = shutil.disk_usage("/")
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": max(1, os.cpu_count() or 1),
        "cpu": lscpu_info(),
        "cuda": cuda_info() if state.device == "cuda" else None,
        "memory": memory_info(),
        "load_average_1m_5m_15m": load_average,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "api_process": process_info(os.getpid()),
        "engine_process": process_info(state.process.pid if state.process else None),
    }


def average_color(image: Image.Image) -> tuple[int, int, int]:
    mean = ImageStat.Stat(image).mean
    return tuple(round(channel) for channel in mean[:3])


def isolate_selection(
    crop: Image.Image,
    shape: Literal["rect", "circle"],
    local_box: tuple[int, int, int, int],
) -> tuple[Image.Image, int]:
    mask = Image.new("L", crop.size)
    draw = ImageDraw.Draw(mask)
    (draw.ellipse if shape == "circle" else draw.rectangle)(local_box, fill=255)
    selected_pixels = mask.histogram()[255]
    background = Image.new("RGB", crop.size, average_color(crop))
    return Image.composite(crop, background, mask), selected_pixels


def normalized_box(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    return [
        round(box[0] / width, 6),
        round(box[1] / height, 6),
        round(box[2] / width, 6),
        round(box[3] / height, 6),
    ]


def prepare_crop(image: Image.Image, region: Region) -> PreparedImage:
    started = perf_counter()
    selection_box = region.pixel_box(image.size)
    crop_box = region.pixel_box(image.size, SETTINGS.crop_padding)
    local_box = (
        selection_box[0] - crop_box[0],
        selection_box[1] - crop_box[1],
        selection_box[2] - crop_box[0],
        selection_box[3] - crop_box[1],
    )
    isolated, selected_pixels = isolate_selection(image.crop(crop_box), region.shape, local_box)

    before_size = isolated.size
    scale = min(1.0, SETTINGS.image_side / max(before_size))
    model_size = (
        max(1, round(before_size[0] * scale)),
        max(1, round(before_size[1] * scale)),
    )
    model_image = (
        isolated
        if model_size == before_size
        else isolated.resize(model_size, RESAMPLING_FILTERS[SETTINGS.resample])
    )

    crop_area = max(1, before_size[0] * before_size[1])
    source_area = max(1, image.width * image.height)
    details = {
        "selection_box": [round(value, 6) for value in region.box],
        "selection_pixels": list(selection_box),
        "crop_box": normalized_box(crop_box, image.size),
        "crop_pixels": list(crop_box),
        "source_size": list(image.size),
        "crop_size_before_resize": list(before_size),
        "model_image_size": list(model_size),
        "selected_area_pixels": selected_pixels,
        "crop_area_pixels": crop_area,
        "selected_fraction_of_source": round(selected_pixels / source_area, 6),
        "selected_fraction_of_crop": round(selected_pixels / crop_area, 6),
        "padding_ratio": SETTINGS.crop_padding,
        "resampling": SETTINGS.resample,
        "prepare_ms": round((perf_counter() - started) * 1000, 3),
    }
    return PreparedImage(model_image, details)


def decode_image_bytes(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(BytesIO(data), formats=["JPEG", "PNG", "WEBP"])
        with source:
            if source.width * source.height > SETTINGS.max_image_pixels:
                raise HTTPException(413, "Image has too many decoded pixels")
            if getattr(source, "is_animated", False):
                raise HTTPException(400, "Animated images are not supported")
            return ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(413, "Image has too many decoded pixels") from error
    except (OSError, ValueError) as error:
        raise HTTPException(400, "Invalid or unsupported image") from error


def encode_image(image: Image.Image) -> tuple[str, int, float]:
    started = perf_counter()
    output = BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=SETTINGS.jpeg_quality,
        optimize=False,
    )
    jpeg = output.getvalue()
    encoded = base64.b64encode(jpeg).decode("ascii")
    return (
        f"data:image/jpeg;base64,{encoded}",
        len(jpeg),
        round((perf_counter() - started) * 1000, 3),
    )


Result = TypeVar("Result")


async def run_preprocess(function: Callable[..., Result], *args: Any) -> Result:
    async with PREPROCESS_GATE:
        return await asyncio.to_thread(function, *args)


def clean_label(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else "unknown"
    cleaned = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", first_line.lower())
    cleaned = re.sub(r"^(?:the|an|a)\s+", "", cleaned)
    cleaned = re.split(r"[.!?;:\n]", cleaned, maxsplit=1)[0]
    words = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", cleaned)
    return " ".join(words[:4]) or "unknown"


def token_probability(choice: dict[str, Any]) -> tuple[float | None, list[dict[str, Any]]]:
    content = (choice.get("logprobs") or {}).get("content") or []
    values: list[float] = []
    details: list[dict[str, Any]] = []
    for item in content:
        token = str(item.get("token", ""))
        logprob = item.get("logprob")
        if token and isinstance(logprob, (int, float)):
            value = float(logprob)
            values.append(value)
            details.append({"token": token, "logprob": round(value, 6)})
    if not values:
        return None, details
    return round(math.exp(sum(values) / len(values)), 6), details


def response_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    return ""


def inference_payload(state: EngineState, image_url: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": state.api_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SETTINGS.prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": SETTINGS.max_tokens,
        "stream": False,
    }
    if state.engine == "llama":
        payload["cache_prompt"] = True
    if SETTINGS.logprobs:
        payload.update(logprobs=True, top_logprobs=1)
    return payload


def extract_generation(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices") or []
    if not choices:
        raise HTTPException(502, "Inference engine returned no choices")
    choice = choices[0]
    raw_output = response_text(choice.get("message") or {})
    probability, logprobs = token_probability(choice)
    return {
        "label": clean_label(raw_output),
        "raw_output": raw_output,
        "finish_reason": choice.get("finish_reason"),
        "token_probability": probability,
        "token_logprobs": logprobs,
        "usage": body.get("usage") or {},
        "engine_timings": body.get("timings") or {},
        "response_id": body.get("id"),
    }


async def call_engine(state: EngineState, image_url: str) -> tuple[dict[str, Any], float]:
    if state.client is None:
        raise RuntimeError("Inference HTTP client is not initialized")
    started = perf_counter()
    try:
        response = await state.client.post(
            f"{state.base_url}/chat/completions",
            json=inference_payload(state, image_url),
        )
    except httpx.TimeoutException as error:
        raise HTTPException(504, "Inference engine timed out") from error
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Inference engine request failed: {error}") from error

    engine_http_ms = (perf_counter() - started) * 1000
    if response.is_error:
        message = response.text.strip().replace("\x00", "")[:1200]
        raise HTTPException(
            502,
            f"Inference engine returned HTTP {response.status_code}: {message}",
        )
    try:
        body = response.json()
    except ValueError as error:
        raise HTTPException(502, "Inference engine returned invalid JSON") from error
    return extract_generation(body), round(engine_http_ms, 3)


RESOLVED_DEVICE = resolve_device()
RESOLVED_ENGINE = resolve_engine(RESOLVED_DEVICE)
ENGINE_COMMAND = (
    llama_command(RESOLVED_DEVICE)
    if RESOLVED_ENGINE == "llama"
    else vllm_command()
    if RESOLVED_ENGINE == "vllm"
    else []
)
STATE = EngineState(
    engine=RESOLVED_ENGINE,
    device=RESOLVED_DEVICE,
    base_url=endpoint_url(),
    api_model=SETTINGS.api_model,
    command=ENGINE_COMMAND,
)
INFERENCE_GATE = InferenceGate(SETTINGS.concurrency)
PREPROCESS_GATE = asyncio.Semaphore(SETTINGS.preprocess_concurrency)
IMAGE_SESSIONS = ImageSessionStore(
    SETTINGS.image_session_cache_bytes,
    SETTINGS.max_image_sessions,
    SETTINGS.image_session_ttl,
)


async def no_progress(
    _stage: str,
    _message: str,
    _progress: int,
    _detail: dict[str, Any] | None = None,
) -> None:
    return None


async def classify_pipeline(
    session_id: str,
    selection: Region,
    progress: ProgressCallback = no_progress,
) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    request_started = perf_counter()
    await progress(
        "accepted",
        "Request accepted",
        4,
        {"request_id": request_id, "image_session_id": session_id},
    )

    lookup_started = perf_counter()
    session = await IMAGE_SESSIONS.get(session_id)
    session_lookup_ms = (perf_counter() - lookup_started) * 1000

    await progress("decode", "Decoding staged image", 14, None)
    decode_started = perf_counter()
    source = await run_preprocess(decode_image_bytes, session.data)
    decode_ms = (perf_counter() - decode_started) * 1000
    await progress(
        "decode",
        "Image decoded",
        27,
        {"source_size": list(source.size), "decode_ms": round(decode_ms, 3)},
    )

    await progress("crop", "Preparing selected crop", 36, None)
    try:
        crop = await run_preprocess(prepare_crop, source, selection)
    finally:
        source.close()
    await progress(
        "crop",
        "Crop prepared",
        49,
        {
            "model_image_size": crop.details["model_image_size"],
            "prepare_ms": crop.details["prepare_ms"],
        },
    )

    await progress("encode", "Encoding model image", 55, None)
    try:
        image_url, jpeg_bytes, encode_ms = await run_preprocess(encode_image, crop.image)
    finally:
        crop.image.close()
    await progress(
        "encode",
        "Model image encoded",
        61,
        {"jpeg_bytes": jpeg_bytes, "encode_ms": encode_ms},
    )

    queue_started = perf_counter()
    queue = await INFERENCE_GATE.snapshot()
    queued = queue["active"] >= queue["limit"] or queue["waiting"] > 0
    await progress(
        "queued",
        "Waiting for an inference slot" if queued else "Inference slot available",
        66,
        queue,
    )

    async with INFERENCE_GATE.slot() as running:
        queue_wait_ms = (perf_counter() - queue_started) * 1000
        await progress(
            "inference",
            "Running vision-language inference",
            74,
            {"queue_wait_ms": round(queue_wait_ms, 3), **running},
        )
        generation, engine_http_ms = await call_engine(STATE, image_url)

    request_total_ms = (perf_counter() - request_started) * 1000
    result = {
        "request_id": request_id,
        "image_session_id": session.session_id,
        "label": generation["label"],
        "token_probability": generation["token_probability"],
        "selection": {
            "shape": selection.shape,
            "box": [round(value, 6) for value in selection.box],
        },
        "crop": crop.details,
        "generation": generation,
        "performance": {
            "session_lookup_ms": round(session_lookup_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "crop_ms": crop.details["prepare_ms"],
            "encode_ms": encode_ms,
            "queue_wait_ms": round(queue_wait_ms, 3),
            "engine_http_ms": engine_http_ms,
            "request_total_ms": round(request_total_ms, 3),
            "source_upload_bytes": session.upload_bytes,
            "encoded_jpeg_bytes": jpeg_bytes,
        },
        "runtime": {
            "engine": STATE.engine,
            "device": STATE.device,
            "model": STATE.api_model,
            "threads": SETTINGS.threads if STATE.engine == "llama" else None,
            "concurrency": SETTINGS.concurrency,
        },
    }
    await progress(
        "complete",
        f"Completed: {result['label']}",
        100,
        {"request_total_ms": result["performance"]["request_total_ms"]},
    )
    return result


def engine_info() -> dict[str, Any]:
    return {
        "type": STATE.engine,
        "base_url": STATE.base_url,
        "api_model": STATE.api_model,
        "binary_version": STATE.binary_version,
        "command": STATE.command,
        "startup": STATE.startup,
        "metadata": STATE.metadata,
    }


def configuration_info() -> dict[str, Any]:
    return {
        "threads": SETTINGS.threads,
        "concurrency": SETTINGS.concurrency,
        "preprocess_concurrency": SETTINGS.preprocess_concurrency,
        "request_timeout_seconds": SETTINGS.request_timeout,
        "context_tokens_per_slot": SETTINGS.context_tokens_per_slot,
        "image_tokens": SETTINGS.image_tokens,
        "max_output_tokens": SETTINGS.max_tokens,
        "batch_size": SETTINGS.batch_size,
        "ubatch_size": SETTINGS.ubatch_size,
        "image_side": SETTINGS.image_side,
        "resampling": SETTINGS.resample,
        "jpeg_quality": SETTINGS.jpeg_quality,
        "logprobs_requested": SETTINGS.logprobs,
        "max_upload_bytes": SETTINGS.max_upload_bytes,
        "max_image_pixels": SETTINGS.max_image_pixels,
        "image_session_ttl_seconds": SETTINGS.image_session_ttl,
        "image_session_cache_bytes": SETTINGS.image_session_cache_bytes,
        "max_image_sessions": SETTINGS.max_image_sessions,
    }


async def image_session_sweeper() -> None:
    interval = max(1.0, min(30.0, SETTINGS.image_session_ttl / 2))
    while True:
        await asyncio.sleep(interval)
        await IMAGE_SESSIONS.cleanup()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    await start_engine(STATE)
    cleanup_task = asyncio.create_task(image_session_sweeper())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)
        await IMAGE_SESSIONS.clear()
        await stop_engine(STATE)


APP = FastAPI(title="LFM2.5-VL selected-region classifier", lifespan=lifespan)
WEB_ROOT = Path(__file__).resolve().parent
APP.mount("/assets", StaticFiles(directory=WEB_ROOT / "static"), name="assets")
if SETTINGS.cors_origins:
    APP.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )


@APP.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@APP.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": {
            "native": SETTINGS.model_native,
            "gguf": SETTINGS.model_gguf,
            "api_model": SETTINGS.api_model,
        },
        "engine": engine_info(),
        "device": {
            "requested": SETTINGS.device,
            "resolved": STATE.device,
        },
        "configuration": configuration_info(),
        "hardware": hardware_info(STATE),
        "queue": await INFERENCE_GATE.snapshot(),
        "image_sessions": await IMAGE_SESSIONS.snapshot(),
    }


@APP.post("/image-sessions", status_code=201)
async def create_image_session(
    response: Response,
    image: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    data = await image.read(SETTINGS.max_upload_bytes + 1)
    if len(data) > SETTINGS.max_upload_bytes:
        raise HTTPException(413, "Image is too large")
    if not data:
        raise HTTPException(400, "Image is empty")

    decoded = await run_preprocess(decode_image_bytes, data)
    try:
        width, height = decoded.size
    finally:
        decoded.close()
    session = await IMAGE_SESSIONS.put(data, width, height)

    response.headers["Cache-Control"] = "no-store"
    response.headers["Location"] = f"/image-sessions/{session.session_id}"
    return {
        "session_id": session.session_id,
        "image": {"width": width, "height": height},
        "upload_bytes": session.upload_bytes,
        "expires_in_seconds": SETTINGS.image_session_ttl,
    }


@APP.delete("/image-sessions/{session_id}", status_code=204)
async def delete_image_session(session_id: str) -> Response:
    await IMAGE_SESSIONS.delete(session_id)
    return Response(status_code=204)


@APP.post("/classify")
async def classify(
    session_id: Annotated[str, Form()],
    selection: Annotated[Region, Depends(parse_region)],
) -> dict[str, Any]:
    return await classify_pipeline(session_id, selection)


def ndjson(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


@APP.post("/classify/stream", response_class=StreamingResponse)
async def classify_stream(
    request: Request,
    session_id: Annotated[str, Form()],
    selection: Annotated[Region, Depends(parse_region)],
) -> StreamingResponse:
    """Stream pipeline status events followed by one final result event."""

    async def events() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        started = perf_counter()

        async def report(
            stage: str,
            message: str,
            progress: int,
            detail: dict[str, Any] | None = None,
        ) -> None:
            queue.put_nowait(
                {
                    "event": "status",
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                    "detail": detail,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                }
            )

        async def work() -> None:
            try:
                result = await classify_pipeline(session_id, selection, report)
                queue.put_nowait({"event": "result", "result": result})
            except asyncio.CancelledError:
                raise
            except HTTPException as error:
                queue.put_nowait(
                    {
                        "event": "error",
                        "status": error.status_code,
                        "message": str(error.detail),
                    }
                )
            except Exception as error:  # pragma: no cover - defensive boundary
                LOGGER.exception("Unhandled classification failure")
                queue.put_nowait(
                    {
                        "event": "error",
                        "status": 500,
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                queue.put_nowait({"event": "end"})

        worker = asyncio.create_task(work())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except TimeoutError:
                    if await request.is_disconnected():
                        break
                    continue
                if item["event"] == "end":
                    break
                yield ndjson(item)
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(APP, host="0.0.0.0", port=8000, workers=1)
