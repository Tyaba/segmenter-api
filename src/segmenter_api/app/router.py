from functools import partial

from fastapi import APIRouter, Depends

from segmenter_api.di import resolve
from segmenter_api.usecase.ui.text2segment import (
    Text2SegmentRequest,
    Text2SegmentResponse,
    SegmenterUserInterface,
)

router = APIRouter()


@router.post("/text2segment")
def text2segment(
    request: Text2SegmentRequest,
    segmenter_user_interface: SegmenterUserInterface = Depends(
        partial(resolve, SegmenterUserInterface)
    ),
) -> Text2SegmentResponse:
    return segmenter_user_interface.text2segment(request)
