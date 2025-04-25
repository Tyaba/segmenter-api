"""uv run python src/segmenter_api/usecase/service/text2segment.py \
--text "plastic bottle" \
--image-path tests/data/abema_water.png \
--output-image-dir data/abema_water_segmented
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

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
                image=text2segment_input.image.convert("RGB"),
            )
        )
        if len(text2bbox_output.bboxes) == 0:
            return Text2SegmentOutput(
                masks=[],
                labels=[],
                text2bbox_output=text2bbox_output,
                bbox2segment_output=Bbox2SegmentOutput(masks=[]),
            )
        assert_bboxes_in_image(
            bboxes=text2bbox_output.bboxes,
            image_size=text2segment_input.image.size,
        )
        bbox2segment_output: Bbox2SegmentOutput = segmenter.bbox2segment(
            bbox2segment_input=Bbox2SegmentInput(
                image=text2segment_input.image.convert("RGB"),
                bboxes=text2bbox_output.bboxes,
            )
        )
        masks = bbox2segment_output.masks
        assert_mask_size_is_image_size(
            masks=masks,
            image_size=text2segment_input.image.size,
        )
        return Text2SegmentOutput(
            masks=masks,
            labels=text2bbox_output.labels,
            text2bbox_output=text2bbox_output,
            bbox2segment_output=bbox2segment_output,
        )


def assert_bboxes_in_image(
    bboxes: list[tuple[float, float, float, float]], image_size: tuple[int, int]
) -> None:
    assert all(
        bbox[2] < image_size[0] and bbox[3] < image_size[1] for bbox in bboxes
    ), (
        "bboxの座標がinput_imageのサイズを超えています。"
        f"input_image: {image_size}, "
        f"bboxes: {bboxes}"
    )


def assert_mask_size_is_image_size(
    masks: list[Image.Image], image_size: tuple[int, int]
) -> None:
    assert all(mask.size == image_size for mask in masks), (
        "maskのサイズがinput_imageのサイズと一致しません。"
        f"input_image: {image_size}, "
        f"masks: {[mask.size for mask in masks]}"
    )


def main(args: Namespace) -> None:
    from segmenter_api.di import resolve

    input_image = Image.open(args.image_path)
    output = resolve(Text2SegmentUsecase).text2segment(
        Text2SegmentInput(
            texts=[args.text],
            image=input_image,
            detector_type=DetectorType.FLORENCE2_BASE,
            segmenter_type=SegmenterType.SAM2,
        )
    )
    args.output_image_dir.mkdir(parents=True, exist_ok=True)
    for i, (mask, label) in enumerate(zip(output.masks, output.labels, strict=True)):
        cropped = input_image.copy().convert("RGBA")
        cropped.putalpha(mask)
        cropped.save(args.output_image_dir / f"{i}_{label}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
