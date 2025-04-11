from injector import inject
from PIL import Image
from pydantic import BaseModel, ConfigDict, field_serializer
from typing import Self
from segmenter_api.domain.factory.detector_factory import DetectorType
from segmenter_api.domain.factory.segmenter_factory import SegmenterType
from segmenter_api.usecase.service.text2segment import (
    Text2SegmentInput,
    Text2SegmentOutput,
    Text2SegmentUsecase,
)
from segmenter_api.utils.image import base642pil, image2boolean, boolean2image


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
    masks: list[list[list[bool]]]

    @classmethod
    def from_images(cls, images: list[Image.Image]) -> Self:
        return cls(masks=[image2boolean(image) for image in images])

    @property
    def mask_images(self) -> list[Image.Image]:
        return [boolean2image(mask) for mask in self.masks]


class SegmenterUserInterface:
    @inject
    def __init__(self, text2segment_usecase: Text2SegmentUsecase):
        self.text2segment_usecase = text2segment_usecase

    def text2segment(
        self, text2segment_request: Text2SegmentRequest
    ) -> Text2SegmentResponse:
        usecase_input = Text2SegmentInput(
            texts=text2segment_request.texts,
            image=base642pil(text2segment_request.image),
            detector_type=text2segment_request.detector_type,
            segmenter_type=text2segment_request.segmenter_type,
        )
        usecase_output = self.text2segment_usecase.text2segment(
            text2segment_input=usecase_input,
        )
        return Text2SegmentResponse.from_images(usecase_output.masks)
