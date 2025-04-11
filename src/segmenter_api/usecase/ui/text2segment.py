from injector import inject
from pydantic import BaseModel, field_serializer

from segmenter_api.domain.factory.detector_factory import DetectorType
from segmenter_api.domain.factory.segmenter_factory import SegmenterType
from segmenter_api.usecase.service.text2segment import Text2SegmentUsecase


class Text2SegmentRequest(BaseModel):
    detector_type: DetectorType
    segmenter_type: SegmenterType
    texts: list[str] | None = None

    @field_serializer("detector_type")
    def serialize_detector_type(self, detector_type: DetectorType) -> str:
        return detector_type.value

    @field_serializer("segmenter_type")
    def serialize_segmenter_type(self, segmenter_type: SegmenterType) -> str:
        return segmenter_type.value


class Text2SegmentResponse(BaseModel):
    segments: list[list[float]]


class SegmenterUserInterface:
    @inject
    def __init__(self, text2segment_usecase: Text2SegmentUsecase):
        self.text2segment_usecase = text2segment_usecase

    def text2segment(self, text2segment_request: Text2SegmentRequest) -> Text2SegmentResponse:
        text2segment_output = self.text2segment_usecase.text2segment(text2segment_request)
        return Text2SegmentResponse(segments=text2segment_output.segments)
