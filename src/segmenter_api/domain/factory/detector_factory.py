from abc import ABC, abstractmethod
from enum import Enum

from segmenter_api.domain.service.detector import Detector


class DetectorType(Enum):
    FLORENCE2_BASE = "florence2_base"
    FLORENCE2_LARGE = "florence2_large"
    GROUNDING_DINO = "grounding_dino"


class DetectorFactoryInterface(ABC):
    @abstractmethod
    def create(self, detector_type: DetectorType) -> Detector:
        raise NotImplementedError
