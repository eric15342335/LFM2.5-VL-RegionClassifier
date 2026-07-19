# LFM2.5-VL selected-region classifier

A single FastAPI service provides the inference API and the mobile-friendly diagnostic UI.
The browser consumes newline-delimited JSON from `POST /classify/stream` and reports decode,
crop, encode, queue, inference, and completion stages while the request is running.

## Install and run

Install a current `llama-server` separately and ensure it is on `PATH`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
set -a; . ./.env; set +a
python backend.py
```

Open port 8000. Run only one Uvicorn worker when the backend launches the engine; additional
workers would each launch another model server.

## Deployment defaults

- `THREADS=3` and `CONCURRENCY=2`.
- `Q4_K_M` remains the default because the supplied benchmark indicates prompt processing
  dominates this short-output workload.
- `REQUEST_TIMEOUT=300` covers backend-to-engine HTTP requests. Configure reverse-proxy
  timeouts separately.
- Bilinear downsampling is used by default. Small crops are not upscaled.
- The backend does not force `--mlock` or `--no-mmap`. Add tested llama.cpp flags through
  `LLAMA_SERVER_EXTRA_ARGS` when required by a specific host.
- No frontend build step is required; source files are served directly for easier debugging.

## Development checks

These are optional development checks and are not part of installation or runtime startup.

```bash
pip install -r requirements-dev.txt
isort --check-only backend.py tests
ruff check .
ruff format --check .
pytest -q

npm ci
npm run check
```
