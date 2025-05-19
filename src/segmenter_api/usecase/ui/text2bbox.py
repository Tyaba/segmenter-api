from injector import inject

from segmenter_api.domain.model.text2bbox import (
    Text2BboxInput,
    Text2BboxRequest,
    Text2BboxResponse,
)
from segmenter_api.usecase.service.text2bbox import Text2BboxUsecase
from segmenter_api.utils.image import base642pil
from segmenter_api.utils.time import stop_watch


class Text2BboxUserInterface:
    @inject
    def __init__(self, text2bbox_usecase: Text2BboxUsecase):
        self.text2bbox_usecase = text2bbox_usecase

    @stop_watch
    def text2bbox(self, text2bbox_request: Text2BboxRequest) -> Text2BboxResponse:
        usecase_input = Text2BboxInput(
            texts=text2bbox_request.texts,
            image=base642pil(image_base64=text2bbox_request.image),
            detector_type=text2bbox_request.detector_type,
        )
        usecase_output = self.text2bbox_usecase.text2bbox(
            text2bbox_input=usecase_input,
        )
        return Text2BboxResponse(
            labels=usecase_output.labels,
            bboxes=[list(bbox) for bbox in usecase_output.bboxes],
        )
