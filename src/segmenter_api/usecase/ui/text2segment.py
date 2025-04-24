from injector import inject

from segmenter_api.domain.model.text2segment import (
    Text2SegmentInput,
    Text2SegmentRequest,
    Text2SegmentResponse,
)
from segmenter_api.usecase.service.text2segment import Text2SegmentUsecase
from segmenter_api.utils.image import base642pil, pil2base64
from segmenter_api.utils.time import stop_watch


class Text2SegmentUserInterface:
    @inject
    def __init__(self, text2segment_usecase: Text2SegmentUsecase):
        self.text2segment_usecase = text2segment_usecase

    @stop_watch
    def text2segment(
        self, text2segment_request: Text2SegmentRequest
    ) -> Text2SegmentResponse:
        usecase_input = Text2SegmentInput(
            texts=text2segment_request.texts,
            image=base642pil(image_base64=text2segment_request.image),
            detector_type=text2segment_request.detector_type,
            segmenter_type=text2segment_request.segmenter_type,
        )
        usecase_output = self.text2segment_usecase.text2segment(
            text2segment_input=usecase_input,
        )
        return Text2SegmentResponse(
            labels=usecase_output.labels,
            masks=[pil2base64(image=mask) for mask in usecase_output.masks],
            bboxes=[list(bbox) for bbox in usecase_output.text2bbox_output.bboxes],
        )
