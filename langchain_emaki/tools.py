"""emaki video and image generation as LangChain tools.

emaki (https://emaki.ai) is the Japanese-language AI video generator for
Seedance 2.5 and Seedance 2.0. Both tools run on an emaki account: pass
``api_key`` or set ``EMAKI_API_KEY`` (create a key at
https://emaki.ai/settings/apikeys). Runs cost credits on that account, the
same price as on the website.
"""

from typing import Any, Dict, Literal, Optional, Type

import emaki
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def _explain(e: emaki.EmakiError) -> str:
    # Hand refusals back to the agent as text so it can react (wait, reword,
    # pick another model, ask for a key) instead of crashing the run.
    if isinstance(e, emaki.AuthError):
        return f"emaki needs a valid API key (create one at https://emaki.ai/settings/apikeys): {e}"
    if isinstance(e, emaki.QuotaError):
        return f"The emaki account is out of credits: {e}"
    if isinstance(e, emaki.PlanError):
        return f"That model needs a purchase on the emaki account first; try seedance_2_mini: {e}"
    if isinstance(e, emaki.BusyError):
        return f"Another emaki run on this account is still generating; wait for it: {e}"
    if isinstance(e, emaki.RejectedError):
        return f"emaki refused the prompt, reword it: {e}"
    if isinstance(e, emaki.GenerationFailed):
        return f"The emaki run failed ({e.code or 'no failure code'}); reword the prompt rather than retrying it as is"
    return f"emaki error: {e}"


def _params(**kw: Any) -> Dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


class EmakiVideoInput(BaseModel):
    """Arguments for :class:`EmakiVideoGenerator`."""

    prompt: str = Field(description="The shot: what happens, where, and how the camera moves.")
    image_url: Optional[str] = Field(default=None, description="Public https url of a first frame to animate (image to video).")
    duration: Optional[int] = Field(default=None, description="Clip length in seconds. Seedance 2.5 takes 4 to 30, default 5.")
    resolution: Optional[Literal["480p", "720p", "1080p"]] = Field(default=None, description="Output resolution, default 480p.")
    aspect_ratio: Optional[Literal["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]] = Field(
        default=None, description="Shape of the clip, default 16:9."
    )


class EmakiVideoGenerator(BaseTool):
    """Generate a video clip on emaki and return its url. Takes minutes.

    ``model`` defaults to ``seedance_2_5``; other ids: ``seedance_2``,
    ``seedance_2_fast``, ``seedance_2_mini``, ``minimax_h3``, ``veo_3_1``.

    Example:
        .. code-block:: python

            from langchain_emaki import EmakiVideoGenerator

            tool = EmakiVideoGenerator()  # reads EMAKI_API_KEY
            tool.invoke({"prompt": "a paper boat drifting down a rain gutter, low angle", "duration": 5})
    """

    name: str = "emaki_video_generator"
    description: str = (
        "Generates a short video clip with Seedance 2.5 from a text prompt, or animates "
        "an image at a public url, and returns a url to the clip. Takes several minutes."
    )
    args_schema: Type[BaseModel] = EmakiVideoInput
    api_key: Optional[str] = None
    model: str = emaki.DEFAULT_VIDEO_MODEL
    generate_audio: Optional[bool] = None
    timeout: float = 900.0

    def _run(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        **_: object,
    ) -> str:
        try:
            return emaki.generate_video(
                prompt,
                self.model,
                api_key=self.api_key,
                timeout=self.timeout,
                **_params(
                    image=image_url,
                    duration=duration,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    generate_audio=self.generate_audio,
                ),
            )
        except emaki.EmakiError as e:
            return _explain(e)


class EmakiImageInput(BaseModel):
    """Arguments for :class:`EmakiImageGenerator`."""

    prompt: str = Field(
        description="What to draw. Name the light, the material and the composition rather than stacking adjectives."
    )
    aspect_ratio: Optional[Literal["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "21:9"]] = Field(
        default=None, description="Shape of the image, default 1:1."
    )


class EmakiImageGenerator(BaseTool):
    """Generate an image on emaki and return its url.

    ``model`` defaults to ``nano_banana_2_lite``, the cheapest; other ids:
    ``nano_banana_2``, ``seedream_5_lite``, ``gpt_image_2``. A generated image
    makes a good first frame for :class:`EmakiVideoGenerator`.
    """

    name: str = "emaki_image_generator"
    description: str = "Generates an image from a text prompt and returns a url to the result."
    args_schema: Type[BaseModel] = EmakiImageInput
    api_key: Optional[str] = None
    model: str = emaki.DEFAULT_IMAGE_MODEL
    timeout: float = 900.0

    def _run(self, prompt: str, aspect_ratio: Optional[str] = None, **_: object) -> str:
        try:
            return emaki.generate_image(
                prompt,
                self.model,
                api_key=self.api_key,
                timeout=self.timeout,
                **_params(aspect_ratio=aspect_ratio),
            )
        except emaki.EmakiError as e:
            return _explain(e)
