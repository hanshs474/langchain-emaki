# langchain-emaki

LangChain tools for emaki (emaki.ai), the Japanese-language AI video generator for Seedance 2.5 and Seedance 2.0. Give an agent [Seedance 2.5](https://emaki.ai/seedance-2-5) text to video, [image to video](https://emaki.ai/image-to-video) and AI images on one [emaki](https://emaki.ai/?utm_source=pypi-langchain&utm_medium=package) API key.

```bash
pip install langchain-emaki
export EMAKI_API_KEY=sk-...   # create one at https://emaki.ai/settings/apikeys
```

```python
from langchain_emaki import EmakiImageGenerator, EmakiVideoGenerator

video = EmakiVideoGenerator()   # Seedance 2.5 by default
image = EmakiImageGenerator()   # nano_banana_2_lite by default

video.invoke({"prompt": "a paper boat drifting down a rain gutter, low angle, morning light", "duration": 5})
image.invoke({"prompt": "a paper-cut layered mountain range at dusk", "aspect_ratio": "16:9"})
```

Hand both to any tool-calling agent:

```python
from langchain.agents import create_agent

agent = create_agent("openai:gpt-5", tools=[image, video])
agent.invoke({"messages": [("user", "Make a 16:9 still of a lighthouse in fog, then a 5-second clip that starts from it")]})
```

## Tools

| Tool | Input | Returns |
|---|---|---|
| `emaki_video_generator` | `prompt`, optional `image_url` (first frame), `duration`, `resolution`, `aspect_ratio` | URL of the clip |
| `emaki_image_generator` | `prompt`, optional `aspect_ratio` | URL of the image |

Pass a public https URL as `image_url` and the clip starts from that frame. A generated image works as that frame, which is how the agent above chains the two tools.

Set `model=` on either tool for another engine: `seedance_2`, `seedance_2_fast`, `seedance_2_mini`, `minimax_h3`, `veo_3_1` for video ([Seedance 2.0](https://emaki.ai/seedance-2) is the previous generation); `nano_banana_2`, `seedream_5_lite`, `gpt_image_2` for images. Each run costs credits on your emaki account, the same price as on the website.

Refusals come back to the agent as text rather than an exception: a missing or wrong key, an empty balance, a model that needs a purchase, another run still generating, or a prompt the filter refused. The agent can reword, wait or switch model instead of crashing the run. Video takes minutes; the default timeout is 15 minutes.

Finished clips can be sharpened with the [AI video upscaler](https://emaki.ai/video-upscaler) on the site.

日本語での使い方は [シーダンス](https://emaki.ai/) へ。

Built on the [`emaki`](https://pypi.org/project/emaki/) client. MIT licensed. Made by [emaki](https://emaki.ai/?utm_source=pypi-langchain&utm_medium=package).
