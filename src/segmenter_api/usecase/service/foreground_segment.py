"""uv run python src/segmenter_api/usecase/service/foreground_segment.py \
--image-path tests/data/abema_water.png \
--output-image-path data/abema_water_segmented.png
"""

from argparse import ArgumentParser, Namespace

from injector import inject
from PIL import Image

from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterFactoryInterface,
    SegmenterType,
)
from segmenter_api.domain.model.foreground_segment import (
    ForegroundSegmentUsecaseInput,
    ForegroundSegmentUsecaseOutput,
)
from segmenter_api.domain.service.segmenter import (
    ForegroundSegmentInput,
    ForegroundSegmentOutput,
)
from segmenter_api.utils.time import stop_watch


class ForegroundSegmentUsecase:
    @inject
    def __init__(
        self,
        segmenter_factory: SegmenterFactoryInterface,
    ):
        self.segmenter_factory = segmenter_factory

    @stop_watch
    def foreground_segment(
        self, foreground_segment_usecase_input: ForegroundSegmentUsecaseInput
    ) -> ForegroundSegmentUsecaseOutput:
        segmenter = self.segmenter_factory.create(
            foreground_segment_usecase_input.segmenter_type
        )
        foreground_segment_output: ForegroundSegmentOutput = (
            segmenter.foreground_segment(
                foreground_segment_input=ForegroundSegmentInput(
                    image=foreground_segment_usecase_input.image
                )
            )
        )
        return ForegroundSegmentUsecaseOutput(mask=foreground_segment_output.mask)


def main(args: Namespace) -> None:
    from segmenter_api.di import resolve

    input_image = Image.open(args.image_path)
    output = resolve(ForegroundSegmentUsecase).foreground_segment(
        ForegroundSegmentUsecaseInput(
            image=input_image,
            segmenter_type=SegmenterType.BIREFNET,
        )
    )
    output.mask.save(args.output_image_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output-image-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
