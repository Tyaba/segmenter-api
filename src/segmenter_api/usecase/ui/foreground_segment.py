from injector import inject

from segmenter_api.domain.model.foreground_segment import (
    ForegroundSegmentRequest,
    ForegroundSegmentResponse,
    ForegroundSegmentUsecaseInput,
)
from segmenter_api.usecase.service.foreground_segment import (
    ForegroundSegmentUsecase,
)
from segmenter_api.utils.image import base642pil, image2boolean


class ForegroundSegmentUserInterface:
    @inject
    def __init__(self, foreground_segment_usecase: ForegroundSegmentUsecase):
        self.foreground_segment_usecase = foreground_segment_usecase

    def foreground_segment(
        self, foreground_segment_request: ForegroundSegmentRequest
    ) -> ForegroundSegmentResponse:
        usecase_input = ForegroundSegmentUsecaseInput(
            image=base642pil(image_base64=foreground_segment_request.image),
            segmenter_type=foreground_segment_request.segmenter_type,
        )
        usecase_output = self.foreground_segment_usecase.foreground_segment(
            foreground_segment_usecase_input=usecase_input,
        )
        mask_array = image2boolean(usecase_output.mask)
        return ForegroundSegmentResponse(mask=mask_array)
