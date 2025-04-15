"""uv run python src/segmenter_api/usecase/service/text2segment.py \
--text "plastic bottle" \
--image-path tests/data/abema_water.png \
--output-image-path data/abema_water_segmented.png
"""

from argparse import ArgumentParser, Namespace

from injector import inject
from PIL import Image

from segmenter_api.domain.factory.detector_factory import (
    DetectorFactoryInterface,
    DetectorType,
)
from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterFactoryInterface,
    SegmenterType,
)
from segmenter_api.domain.model.text2segment import (
    Text2SegmentInput,
    Text2SegmentOutput,
)
from segmenter_api.domain.service.detector import Text2BboxInput, Text2BboxOutput
from segmenter_api.domain.service.segmenter import Bbox2SegmentInput, Bbox2SegmentOutput
from segmenter_api.utils.time import stop_watch


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


def main(args: Namespace) -> None:
    from segmenter_api.di import resolve

    input_image = Image.open(args.image_path)
    output = resolve(Text2SegmentUsecase).text2segment(
        Text2SegmentInput(
            texts=[args.text],
            image=input_image,
            detector_type=DetectorType.FLORENCE2,
            segmenter_type=SegmenterType.SAM2,
        )
    )
    output.masks[0].save(args.output_image_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output-image-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
