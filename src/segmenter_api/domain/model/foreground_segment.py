from PIL import Image
from pydantic import BaseModel, ConfigDict, field_serializer

from segmenter_api.domain.factory.segmenter_factory import SegmenterType
from segmenter_api.utils.image import base642pil


class ForegroundSegmentUsecaseInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image
    segmenter_type: SegmenterType


class ForegroundSegmentUsecaseOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mask: Image.Image


class ForegroundSegmentRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str
    segmenter_type: SegmenterType

    @field_serializer("segmenter_type")
    def serialize_segmenter_type(self, segmenter_type: SegmenterType) -> str:
        return segmenter_type.value


class ForegroundSegmentResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mask: str

    @property
    def mask_image(self) -> Image.Image:
        return base642pil(self.mask)
