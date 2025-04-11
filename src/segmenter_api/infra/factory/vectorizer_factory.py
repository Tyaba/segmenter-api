from segmenter_api.domain.factory.segmenter_factory import (
    segmenterFactoryInterface,
    segmenterType,
)
from segmenter_api.domain.service.segmenter import segmenter
from segmenter_api.infra.service.image_segmenter.dreamsim import DreamSim


class segmenterFactory(segmenterFactoryInterface):
    def create(self, segmenter_type: segmenterType) -> segmenter:
        if segmenter_type == segmenterType.IMAGE:
            return DreamSim()
        error_msg = f"Invalid segmenter type: {segmenter_type}"
        raise ValueError(error_msg)
