from PIL import Image
from pydantic import BaseModel, ConfigDict, field_serializer

from segmenter_api.domain.factory.detector_factory import DetectorType
from segmenter_api.domain.factory.segmenter_factory import SegmenterType
from segmenter_api.domain.service.detector import Text2BboxOutput
from segmenter_api.domain.service.segmenter import Bbox2SegmentOutput
from segmenter_api.utils.image import boolean2image


class Text2SegmentInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    texts: list[str]
    image: Image.Image
    detector_type: DetectorType
    segmenter_type: SegmenterType


class Text2SegmentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: list[Image.Image]
    text2bbox_output: Text2BboxOutput
    bbox2segment_output: Bbox2SegmentOutput

    def segment_images(self, image: Image.Image) -> list[Image.Image]:
        return [Image.alpha_composite(image, mask) for mask in self.masks]


class Text2SegmentRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    detector_type: DetectorType
    segmenter_type: SegmenterType
    image: str
    texts: list[str]

    @field_serializer("detector_type")
    def serialize_detector_type(self, detector_type: DetectorType) -> str:
        return detector_type.value

    @field_serializer("segmenter_type")
    def serialize_segmenter_type(self, segmenter_type: SegmenterType) -> str:
        return segmenter_type.value


class Text2SegmentResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: list[list[list[bool]]]

    @property
    def mask_images(self) -> list[Image.Image]:
        return [boolean2image(bool_list=mask) for mask in self.masks]
