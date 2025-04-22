from abc import ABC, abstractmethod
from typing import Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator


class Text2BboxInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    texts: list[str]
    image: Image.Image


class Text2BboxOutput(BaseModel):
    labels: list[str]
    bboxes: list[tuple[float, float, float, float]]

    @model_validator(mode="after")
    def check_labels_and_bboxes(self) -> Self:
        if len(self.labels) != len(self.bboxes):
            error_msg = "labelsとbboxesの長さが一致しません"
            raise ValueError(error_msg)
        return self


class Detector(ABC):
    @abstractmethod
    def text2bbox(self, text2bbox_input: Text2BboxInput) -> Text2BboxOutput:
        raise NotImplementedError
