from unittest import mock

import emaki
from langchain_emaki import EmakiImageGenerator, EmakiVideoGenerator


def test_video_defaults_to_seedance_2_5_and_maps_image():
    with mock.patch.object(emaki, "generate_video", return_value="https://cdn.emaki.ai/v.mp4") as g:
        out = EmakiVideoGenerator(api_key="sk").invoke(
            {"prompt": "a fox", "duration": 6, "image_url": "https://x/y.png", "resolution": "720p"}
        )
    assert out == "https://cdn.emaki.ai/v.mp4"
    assert g.call_args.args[1] == "seedance_2_5"
    assert g.call_args.kwargs["image"] == "https://x/y.png"
    assert g.call_args.kwargs["duration"] == 6
    assert g.call_args.kwargs["resolution"] == "720p"
    assert "aspect_ratio" not in g.call_args.kwargs
    assert g.call_args.kwargs["api_key"] == "sk"


def test_image_defaults_to_cheapest():
    with mock.patch.object(emaki, "generate_image", return_value="https://cdn.emaki.ai/i.png") as g:
        out = EmakiImageGenerator(api_key="sk").invoke({"prompt": "a boat", "aspect_ratio": "16:9"})
    assert out == "https://cdn.emaki.ai/i.png"
    assert g.call_args.args[1] == "nano_banana_2_lite"
    assert g.call_args.kwargs["aspect_ratio"] == "16:9"


def test_refusals_become_text():
    cases = [
        (emaki.AuthError("Invalid API key", 401), "API key"),
        (emaki.QuotaError("Insufficient credits", 402), "out of credits"),
        (emaki.PlanError("purchase required", 403), "purchase"),
        (emaki.BusyError("Video generation is already in progress", 409), "still generating"),
        (emaki.RejectedError("blocked", 400, "content_blocked"), "reword"),
        (emaki.GenerationFailed("failed", None, "timeout"), "timeout"),
    ]
    for err, needle in cases:
        with mock.patch.object(emaki, "generate_video", side_effect=err):
            out = EmakiVideoGenerator(api_key="sk").invoke({"prompt": "x"})
        assert needle in out, (err, out)


def test_real_http_401_goes_to_auth_branch():
    # Exercise the real emaki client with a fake transport returning the live 401 shape.
    def transport(method, url, body, headers):
        return 401, {"error": "Invalid API key. Create one at https://emaki.ai/settings/apikeys"}

    real = emaki.generate_video
    with mock.patch.object(emaki, "generate_video", side_effect=lambda *a, **k: real(*a, transport=transport, **k)):
        out = EmakiVideoGenerator(api_key="sk-bad").invoke({"prompt": "x"})
    assert out.startswith("emaki needs a valid API key")
