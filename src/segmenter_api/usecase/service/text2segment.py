from injector import inject
from pydantic import BaseModel

from src.segmenter_api.domain.factory.detector_factory import DetectorFactoryInterface, DetectorType
from src.segmenter_api.domain.factory.segmenter_factory import SegmenterFactoryInterface, SegmenterType

class Text2SegmentInput(BaseModel):
    texts: list[str]
    detector_type: DetectorType
    segmenter_type: SegmenterType

class Text2SegmentOutput(BaseModel):
    segments: list[list[float]]

class Text2SegmentUsecase:
    @inject
    def __init__(self, segmenter_factory: SegmenterFactoryInterface, detector_factory: DetectorFactoryInterface):
        self.segmenter_factory = segmenter_factory
        self.detector_factory = detector_factory

    def text2segment(self, text2segment_input: Text2SegmentInput) -> Text2SegmentOutput:
        detector = self.detector_factory.create(text2segment_input.detector_type)
        segmenter = self.segmenter_factory.create(text2segment_input.segmenter_type)
        bboxes = detector.detect(text2segment_input.texts)
        segments = segmenter.bbox2segment(bboxes)
        return Text2SegmentOutput(segments=segments)
