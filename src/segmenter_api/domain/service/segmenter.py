from abc import ABC, abstractmethod
from typing import Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)


class Bbox2SegmentInput(BaseModel):
    bboxes: list[list[float]]

class Bbox2SegmentOutput(BaseModel):
    segments: list[list[float]]


class Segmenter(ABC):
    @abstractmethod
    def bbox2segment(self, bbox2segment_input: Bbox2SegmentInput) -> Bbox2SegmentOutput:
        pass
