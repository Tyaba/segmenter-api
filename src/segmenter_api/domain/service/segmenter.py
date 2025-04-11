from abc import ABC, abstractmethod
from typing import Self

from PIL import Image
from pydantic import BaseModel, ConfigDict

from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)


class Bbox2SegmentInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    bboxes: list[list[float]]

class Bbox2SegmentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: list[Image.Image]


class Segmenter(ABC):
    @abstractmethod
    def bbox2segment(self, bbox2segment_input: Bbox2SegmentInput) -> Bbox2SegmentOutput:
        raise NotImplementedError
