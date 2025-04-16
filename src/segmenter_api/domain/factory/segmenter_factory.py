from abc import ABC, abstractmethod
from enum import Enum

from segmenter_api.domain.service.segmenter import Segmenter


class SegmenterType(Enum):
    SAM2 = "sam2"
    BIREFNET = "birefnet"


class SegmenterFactoryInterface(ABC):
    @abstractmethod
    def create(self, segmenter_type: SegmenterType) -> Segmenter:
        raise NotImplementedError
