import asyncio
import json
from dataclasses import replace
from io import BytesIO

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import backend


def test_default_thread_count_is_four() -> None:
    assert backend.SETTINGS.threads == 4
    command = backend.llama_command("cpu")
    assert command[command.index("--threads") + 1] == "4"
    assert command[command.index("--threads-batch") + 1] == "4"
    assert "--mlock" not in command
    assert "--no-mmap" not in command


def test_clean_label() -> None:
    assert backend.clean_label('The "Water Stain".') == "water stain"


def test_region_orders_coordinates() -> None:
    region = backend.Region("rect", 0.8, 0.7, 0.2, 0.1)
    assert region.box == (0.2, 0.1, 0.8, 0.7)


def test_region_rejects_non_finite_coordinates() -> None:
    with pytest.raises(HTTPException, match="finite") as error:
        backend.Region("rect", 0.1, 0.1, float("nan"), 0.9).validate()
    assert error.value.status_code == 400


def test_prepare_crop_downsizes_but_does_not_upscale() -> None:
    small = Image.new("RGB", (100, 80), "white")
    result = backend.prepare_crop(small, backend.Region("rect", 0.2, 0.2, 0.8, 0.8))
    assert result.image.width <= result.details["crop_size_before_resize"][0]
    assert result.image.height <= result.details["crop_size_before_resize"][1]
    assert result.details["resampling"] == "bilinear"

    large = Image.new("RGB", (1600, 1200), "white")
    result = backend.prepare_crop(large, backend.Region("rect", 0.1, 0.1, 0.9, 0.9))
    assert max(result.image.size) <= backend.SETTINGS.image_side


def test_inference_payload_requests_short_deterministic_output() -> None:
    payload = backend.inference_payload(backend.STATE, "data:image/jpeg;base64,abc")
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == backend.SETTINGS.max_tokens
    assert payload["messages"][0]["content"][1]["type"] == "image_url"


def test_image_session_store_refreshes_ttl_and_expires() -> None:
    now = [0.0]
    store = backend.ImageSessionStore(100, 2, 5, clock=lambda: now[0])

    async def scenario() -> None:
        session = await store.put(b"image", 10, 20)
        now[0] = 4
        assert await store.get(session.session_id) is session
        now[0] = 8
        assert (await store.snapshot())["sessions"] == 1
        now[0] = 10
        with pytest.raises(HTTPException, match="missing or expired") as error:
            await store.get(session.session_id)
        assert error.value.status_code == 404

    asyncio.run(scenario())


def test_image_session_store_evicts_least_recently_used_within_limits() -> None:
    store = backend.ImageSessionStore(8, 2, 60)

    async def scenario() -> None:
        first = await store.put(b"1111", 1, 1)
        second = await store.put(b"2222", 1, 1)
        await store.get(first.session_id)
        third = await store.put(b"3333", 1, 1)

        assert await store.get(first.session_id) is first
        assert await store.get(third.session_id) is third
        with pytest.raises(HTTPException) as error:
            await store.get(second.session_id)
        assert error.value.status_code == 404
        assert await store.snapshot() == {
            "sessions": 2,
            "bytes": 8,
            "max_sessions": 2,
            "max_bytes": 8,
            "ttl_seconds": 60,
        }

    asyncio.run(scenario())


def test_image_session_store_enforces_capacity_during_concurrent_writes() -> None:
    store = backend.ImageSessionStore(6, 2, 60)

    async def scenario() -> None:
        await asyncio.gather(*(store.put(bytes([value]) * 3, 1, 1) for value in range(8)))

        assert await store.snapshot() == {
            "sessions": 2,
            "bytes": 6,
            "max_sessions": 2,
            "max_bytes": 6,
            "ttl_seconds": 60,
        }

        with pytest.raises(HTTPException, match="cache capacity") as error:
            await store.put(b"1234567", 1, 1)
        assert error.value.status_code == 413

        await store.clear()
        assert (await store.snapshot())["sessions"] == 0
        assert (await store.snapshot())["bytes"] == 0

    asyncio.run(scenario())


def test_image_session_endpoint_validates_and_stages_upload() -> None:
    output = BytesIO()
    Image.new("RGB", (24, 16), "green").save(output, format="PNG")
    client = TestClient(backend.APP)

    response = client.post(
        "/image-sessions",
        files={"image": ("parrot.png", output.getvalue(), "image/png")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["image"] == {"width": 24, "height": 16}
    assert body["upload_bytes"] == len(output.getvalue())
    assert len(body["session_id"]) >= 40
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["location"].endswith(body["session_id"])
    assert client.delete(f"/image-sessions/{body['session_id']}").status_code == 204


def test_image_session_endpoint_rejects_unsupported_content() -> None:
    client = TestClient(backend.APP)
    response = client.post(
        "/image-sessions",
        files={"image": ("not-an-image.gif", b"not an image", "image/gif")},
    )
    assert response.status_code == 400


def test_image_session_endpoint_enforces_upload_and_pixel_limits(monkeypatch) -> None:
    client = TestClient(backend.APP)
    settings = backend.SETTINGS
    monkeypatch.setattr(backend, "SETTINGS", replace(settings, max_upload_bytes=3))
    response = client.post(
        "/image-sessions",
        files={"image": ("large.png", b"1234", "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Image is too large"

    output = BytesIO()
    Image.new("RGB", (4, 3), "green").save(output, format="PNG")
    monkeypatch.setattr(
        backend,
        "SETTINGS",
        replace(settings, max_upload_bytes=len(output.getvalue()), max_image_pixels=11),
    )
    response = client.post(
        "/image-sessions",
        files={"image": ("too-many-pixels.png", output.getvalue(), "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Image has too many decoded pixels"


def test_expired_session_is_missing_from_regular_and_streaming_endpoints(monkeypatch) -> None:
    now = [0.0]
    store = backend.ImageSessionStore(1_000_000, 2, 5, clock=lambda: now[0])
    monkeypatch.setattr(backend, "IMAGE_SESSIONS", store)
    output = BytesIO()
    Image.new("RGB", (24, 16), "green").save(output, format="PNG")
    client = TestClient(backend.APP)
    session_id = client.post(
        "/image-sessions",
        files={"image": ("parrot.png", output.getvalue(), "image/png")},
    ).json()["session_id"]
    form = {
        "session_id": session_id,
        "shape": "rect",
        "x1": "0.1",
        "y1": "0.1",
        "x2": "0.9",
        "y2": "0.9",
    }

    now[0] = 6
    response = client.post("/classify", data=form)
    assert response.status_code == 404
    assert response.json()["detail"] == "Image session is missing or expired"

    response = client.post("/classify/stream", data=form)
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert [event["event"] for event in events] == ["status", "error"]
    assert events[1] == {
        "event": "error",
        "status": 404,
        "message": "Image session is missing or expired",
    }
    assert asyncio.run(store.snapshot())["sessions"] == 0
    assert client.delete(f"/image-sessions/{session_id}").status_code == 204


def test_classification_reuses_staged_image_without_an_image_field(monkeypatch) -> None:
    output = BytesIO()
    Image.new("RGB", (40, 30), "green").save(output, format="PNG")
    client = TestClient(backend.APP)
    staged = client.post(
        "/image-sessions",
        files={"image": ("parrot.png", output.getvalue(), "image/png")},
    ).json()

    async def fake_call_engine(_state, _image_url):
        return (
            {
                "label": "parrot",
                "token_probability": 0.9,
                "raw_output": "parrot",
            },
            1.5,
        )

    monkeypatch.setattr(backend, "call_engine", fake_call_engine)
    form = {
        "session_id": staged["session_id"],
        "shape": "rect",
        "x1": "0.1",
        "y1": "0.1",
        "x2": "0.9",
        "y2": "0.9",
    }
    first = client.post("/classify", data=form)
    second = client.post("/classify", data=form)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["label"] == "parrot"
    assert second.json()["image_session_id"] == staged["session_id"]
    assert "upload_read_ms" not in second.json()["performance"]
    assert client.delete(f"/image-sessions/{staged['session_id']}").status_code == 204


def test_classification_stream_returns_statuses_and_reuses_staged_image(monkeypatch) -> None:
    output = BytesIO()
    Image.new("RGB", (40, 30), "green").save(output, format="PNG")
    client = TestClient(backend.APP)
    staged = client.post(
        "/image-sessions",
        files={"image": ("parrot.png", output.getvalue(), "image/png")},
    ).json()

    async def fake_call_engine(_state, _image_url):
        return (
            {
                "label": "parrot",
                "token_probability": 0.9,
                "raw_output": "parrot",
            },
            1.5,
        )

    monkeypatch.setattr(backend, "call_engine", fake_call_engine)
    response = client.post(
        "/classify/stream",
        data={
            "session_id": staged["session_id"],
            "shape": "rect",
            "x1": "0.1",
            "y1": "0.1",
            "x2": "0.9",
            "y2": "0.9",
        },
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert response.headers["x-accel-buffering"] == "no"
    assert events[-1]["event"] == "result"
    assert events[-1]["result"]["label"] == "parrot"
    assert events[-1]["result"]["image_session_id"] == staged["session_id"]
    assert {event["stage"] for event in events[:-1]} >= {
        "accepted",
        "decode",
        "crop",
        "encode",
        "queued",
        "inference",
    }
    assert client.delete(f"/image-sessions/{staged['session_id']}").status_code == 204
