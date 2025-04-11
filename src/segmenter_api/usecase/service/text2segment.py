from injector import inject
from PIL import Image
from pydantic import BaseModel, ConfigDict

from segmenter_api.domain.factory.detector_factory import (
    DetectorFactoryInterface,
    DetectorType,
)
from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterFactoryInterface,
    SegmenterType,
)
from segmenter_api.domain.service.detector import Text2BboxInput, Text2BboxOutput
from segmenter_api.domain.service.segmenter import Bbox2SegmentInput, Bbox2SegmentOutput
from segmenter_api.utils.time import stop_watch

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


class Text2SegmentUsecase:
    @inject
    def __init__(
        self,
        segmenter_factory: SegmenterFactoryInterface,
        detector_factory: DetectorFactoryInterface,
    ):
        self.segmenter_factory = segmenter_factory
        self.detector_factory = detector_factory

    @stop_watch
    def text2segment(self, text2segment_input: Text2SegmentInput) -> Text2SegmentOutput:
        detector = self.detector_factory.create(text2segment_input.detector_type)
        segmenter = self.segmenter_factory.create(text2segment_input.segmenter_type)
        text2bbox_output: Text2BboxOutput = detector.text2bbox(
            text2bbox_input=Text2BboxInput(
                texts=text2segment_input.texts,
                image=text2segment_input.image,
            )
        )
        bbox2segment_output: Bbox2SegmentOutput = segmenter.bbox2segment(
            bbox2segment_input=Bbox2SegmentInput(
                image=text2segment_input.image,
                bboxes=text2bbox_output.bboxes,
            )
        )
        masks = bbox2segment_output.masks
        return Text2SegmentOutput(
            masks=masks,
            text2bbox_output=text2bbox_output,
            bbox2segment_output=bbox2segment_output,
        )
