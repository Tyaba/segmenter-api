from functools import partial

from fastapi import APIRouter, Depends

from segmenter_api.di import resolve
from segmenter_api.usecase.ui.foreground_segment import (
    ForegroundSegmentRequest,
    ForegroundSegmentResponse,
    ForegroundSegmentUserInterface,
)
from segmenter_api.usecase.ui.text2segment import (
    Text2SegmentRequest,
    Text2SegmentResponse,
    Text2SegmentUserInterface,
)
from segmenter_api.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/text2segment", response_model=Text2SegmentResponse)
def text2segment(
    request: Text2SegmentRequest,
    text2segment_user_interface: Text2SegmentUserInterface = Depends(
        partial(resolve, Text2SegmentUserInterface)
    ),
) -> Text2SegmentResponse:
    return text2segment_user_interface.text2segment(request)


@router.post("/foreground_segment", response_model=ForegroundSegmentResponse)
def foreground_segment(
    request: ForegroundSegmentRequest,
    foreground_segment_user_interface: ForegroundSegmentUserInterface = Depends(
        partial(resolve, ForegroundSegmentUserInterface)
    ),
) -> ForegroundSegmentResponse:
    return foreground_segment_user_interface.foreground_segment(request)
