from PIL import Image

import backend


def test_default_thread_count_is_three() -> None:
    assert backend.SETTINGS.threads == 3
    command = backend.llama_command("cpu")
    assert command[command.index("--threads") + 1] == "3"
    assert command[command.index("--threads-batch") + 1] == "3"
    assert "--mlock" not in command
    assert "--no-mmap" not in command


def test_clean_label() -> None:
    assert backend.clean_label('The "Water Stain".') == "water stain"


def test_region_orders_coordinates() -> None:
    region = backend.Region("rect", 0.8, 0.7, 0.2, 0.1)
    assert region.box == (0.2, 0.1, 0.8, 0.7)


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
