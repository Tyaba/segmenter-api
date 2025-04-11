from abc import ABC, abstractmethod
from enum import Enum

from src.segmenter_api.domain.service.segmenter import Segmenter


class SegmenterType(Enum):
    SAM2 = "sam2"


class SegmenterFactoryInterface(ABC):
    @abstractmethod
    def create(self, segmenter_type: SegmenterType) -> Segmenter:
        pass
