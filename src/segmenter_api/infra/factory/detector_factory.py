from functools import lru_cache

from segmenter_api.domain.factory.detector_factory import (
    DetectorFactoryInterface,
    DetectorType,
)
from segmenter_api.domain.repository.file import FileRepositoryInterface
from segmenter_api.domain.service.detector import Detector
from segmenter_api.infra.service.detector.florence2_detector import (
    Florence2Detector,
)
from segmenter_api.infra.service.detector.grounding_dino import (
    GroundingDinoDetector,
)
from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)


class DetectorFactory(DetectorFactoryInterface):
    @lru_cache
    def create(self, detector_type: DetectorType) -> Detector:
        from segmenter_api.di import resolve

        logger.info(f"Creating detector: {detector_type}")
        if detector_type == DetectorType.FLORENCE2_BASE:
            return Florence2Detector(
                file_repository=resolve(FileRepositoryInterface),
                model_type="base",
            )
        elif detector_type == DetectorType.FLORENCE2_LARGE:
            return Florence2Detector(
                file_repository=resolve(FileRepositoryInterface),
                model_type="large",
            )
        elif detector_type == DetectorType.GROUNDING_DINO:
            return resolve(GroundingDinoDetector)
        error_msg = f"Invalid detector type: {detector_type}"
        raise ValueError(error_msg)
