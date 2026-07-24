# LFM2.5-VL selected-region classifier

A single FastAPI service provides the inference API and the mobile-friendly diagnostic UI.
The browser uploads each selected image once to `POST /image-sessions`, then sends only the
opaque session ID and normalized crop coordinates to `POST /classify/stream`. The streaming
response reports decode, crop, encode, queue, inference, and completion stages.

## Install and run

Install a current `llama-server` separately and ensure it is on `PATH`.

```bash
uv sync
cp .env.example .env
set -a; . ./.env; set +a
python backend.py
```

Open port 8000. Run only one Uvicorn worker when the backend launches the engine; additional
workers would each launch another model server.

## Deployment defaults

- `THREADS=4` and `CONCURRENCY=2`.
- `Q4_K_M` is the default model quantization.
- `REQUEST_TIMEOUT=300` covers backend-to-engine HTTP requests. Configure reverse-proxy
  timeouts separately.
- Bilinear downsampling is used by default. Small crops are not upscaled.
- Uploaded image bytes are held in a process-local, compressed-byte LRU cache. Sessions use a
  sliding 15-minute TTL and are bounded to 32 entries / 256 MiB by default.
- `MAX_BYTES` limits encoded upload size and `MAX_IMAGE_PIXELS` limits decoded dimensions.
- Session IDs are bearer capabilities. Bind them to authenticated users before exposing this
  demo as a multi-user public service.
- Process-local sessions require one API worker. Use a shared Redis or object-storage
  implementation behind the same API before horizontally scaling.
- The backend does not force `--mlock` or `--no-mmap`. Add tested llama.cpp flags through
  `LLAMA_SERVER_EXTRA_ARGS` when required by a specific host.
- No frontend build step is required; source files are served directly for easier debugging.

## Development checks

These are optional development checks and are not part of installation or runtime startup.

```bash
uv sync
uv run isort --check-only backend.py tests
uv run ruff check .
uv run ruff format --check .
uv run pytest -q

pnpm install
pnpm run check
```
