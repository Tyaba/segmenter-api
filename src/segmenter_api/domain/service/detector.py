from abc import ABC, abstractmethod

from PIL import Image
from pydantic import BaseModel, ConfigDict


class Text2BboxInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    texts: list[str]
    image: Image.Image


class Text2BboxOutput(BaseModel):
    bboxes: list[list[float]]


class Detector(ABC):
    @abstractmethod
    def text2bbox(self, text2bbox_input: Text2BboxInput) -> Text2BboxOutput:
        raise NotImplementedError
