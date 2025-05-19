from abc import ABC, abstractmethod
from typing import Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator


class DetectorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    texts: list[str]
    image: Image.Image


class DetectorOutput(BaseModel):
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
    def detect(self, detector_input: DetectorInput) -> DetectorOutput:
        raise NotImplementedError
