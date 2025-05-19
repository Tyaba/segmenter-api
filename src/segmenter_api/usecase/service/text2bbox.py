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
from segmenter_api.domain.model.text2bbox import Text2BboxInput, Text2BboxOutput
from segmenter_api.domain.service.detector import DetectorInput, DetectorOutput
from segmenter_api.utils.time import stop_watch


class Text2BboxUsecase:
    @inject
    def __init__(
        self,
        detector_factory: DetectorFactoryInterface,
    ):
        self.detector_factory = detector_factory

    @stop_watch
    def text2bbox(self, text2bbox_input: Text2BboxInput) -> Text2BboxOutput:
        detector = self.detector_factory.create(text2bbox_input.detector_type)
        detector_output: DetectorOutput = detector.detect(
            detector_input=DetectorInput(
                texts=text2bbox_input.texts,
                image=text2bbox_input.image.convert("RGB"),
            )
        )
        assert_bboxes_in_image(
            bboxes=detector_output.bboxes,
            image_size=text2bbox_input.image.size,
        )
        return Text2BboxOutput(
            bboxes=detector_output.bboxes,
            labels=detector_output.labels,
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
    output = resolve(Text2BboxUsecase).text2segment(
        Text2BboxInput(
            texts=[args.text],
            image=input_image,
            detector_type=DetectorType.FLORENCE2_BASE,
        )
    )
    args.output_image_dir.mkdir(parents=True, exist_ok=True)
    for i, (bbox, label) in enumerate(zip(output.bboxes, output.labels, strict=True)):
        print(f"{i}: {label}, {bbox}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
