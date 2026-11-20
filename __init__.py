from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .Audio8 import (
    Audio8TTSLoadModel, Audio8TTSFromModel,
    Audio8ASRLoadModel, Audio8ASRFromModel
)


class Audio8Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Audio8TTSLoadModel,
            Audio8TTSFromModel,
            Audio8ASRLoadModel,
            Audio8ASRFromModel,
        ]


# ComfyUI calls this to load your extension and its nodes.
async def comfy_entrypoint() -> Audio8Extension:
    return Audio8Extension()
