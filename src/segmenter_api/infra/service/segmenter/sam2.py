from segmenter_api.domain.service.segmenter import (
    Bbox2SegmentInput,
    Bbox2SegmentOutput,
    Segmenter,
)


class SAM2(Segmenter):
    def __init__(self):
        pass

    def bbox2segment(self, bbox2segment_input: Bbox2SegmentInput) -> Bbox2SegmentOutput:
        pass
