from functools import lru_cache

from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterFactoryInterface,
    SegmenterType,
)
from segmenter_api.domain.service.segmenter import Segmenter
from segmenter_api.infra.service.segmenter.birefnet import BiRefNet
from segmenter_api.infra.service.segmenter.sam2 import SAM2


class SegmenterFactory(SegmenterFactoryInterface):
    @lru_cache
    def create(self, segmenter_type: SegmenterType) -> Segmenter:
        from segmenter_api.di import resolve

        if segmenter_type == SegmenterType.SAM2:
            return resolve(SAM2)
        if segmenter_type == SegmenterType.BIREFNET:
            return resolve(BiRefNet)
        error_msg = f"Invalid segmenter type: {segmenter_type}"
        raise ValueError(error_msg)
