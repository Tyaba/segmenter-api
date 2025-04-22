from abc import ABC, abstractmethod

from PIL import Image
from pydantic import BaseModel, ConfigDict

from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)


class Bbox2SegmentInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    bboxes: list[tuple[float, float, float, float]]


class Bbox2SegmentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: list[Image.Image]


class ForegroundSegmentInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image


class ForegroundSegmentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mask: Image.Image


class Segmenter(ABC):
    @abstractmethod
    def bbox2segment(self, bbox2segment_input: Bbox2SegmentInput) -> Bbox2SegmentOutput:
        raise NotImplementedError

    @abstractmethod
    def foreground_segment(
        self, foreground_segment_input: ForegroundSegmentInput
    ) -> ForegroundSegmentOutput:
        raise NotImplementedError
