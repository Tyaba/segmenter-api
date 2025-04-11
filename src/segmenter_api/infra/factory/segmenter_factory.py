from functools import lru_cache

from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterFactoryInterface,
    SegmenterType,
)
from segmenter_api.domain.service.segmenter import Segmenter
from segmenter_api.infra.service.segmenter.sam2 import SAM2


class SegmenterFactory(SegmenterFactoryInterface):
    @lru_cache
    def create(self, segmenter_type: SegmenterType) -> Segmenter:
        if segmenter_type == SegmenterType.SAM2:
            return SAM2()
        error_msg = f"Invalid segmenter type: {segmenter_type}"
        raise ValueError(error_msg)
