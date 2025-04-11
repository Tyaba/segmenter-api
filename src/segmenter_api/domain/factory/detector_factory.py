from abc import ABC, abstractmethod
from enum import Enum

from segmenter_api.domain.service.detector import Detector


class DetectorType(Enum):
    FLORENCE2 = "florence2"


class DetectorFactoryInterface(ABC):
    @abstractmethod
    def create(self, detector_type: DetectorType) -> Detector:
        raise NotImplementedError
