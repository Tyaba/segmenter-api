from pathlib import Path

import pytest
from PIL import Image
from pytest_mock import MockFixture

from segmenter_api.domain.factory.detector_factory import DetectorFactoryInterface
from segmenter_api.domain.factory.segmenter_factory import SegmenterFactoryInterface
from segmenter_api.domain.service.detector import Detector, DetectorOutput
from segmenter_api.domain.service.segmenter import (
    Bbox2SegmentOutput,
    ForegroundSegmentOutput,
    Segmenter,
)


@pytest.fixture
def mock_detector(mocker: MockFixture) -> Detector:
    detector = mocker.Mock(spec=Detector)
    detector.detect.return_value = DetectorOutput(
        bboxes=[(0, 0, 50, 50)], labels=["test object"]
    )
    return detector


@pytest.fixture
def mock_segmenter(mocker: MockFixture) -> Segmenter:
    segmenter = mocker.Mock(spec=Segmenter)
    segmenter.bbox2segment.return_value = Bbox2SegmentOutput(
        masks=[Image.new("RGB", (100, 100))]
    )
    segmenter.foreground_segment.return_value = ForegroundSegmentOutput(
        mask=Image.new("RGB", (100, 100))
    )
    return segmenter


@pytest.fixture
def mock_detector_factory(
    mocker: MockFixture, mock_detector: Detector
) -> DetectorFactoryInterface:
    factory = mocker.Mock(spec=DetectorFactoryInterface)
    factory.create.return_value = mock_detector
    return factory


@pytest.fixture
def mock_segmenter_factory(
    mocker: MockFixture, mock_segmenter: Segmenter
) -> SegmenterFactoryInterface:
    factory = mocker.Mock(spec=SegmenterFactoryInterface)
    factory.create.return_value = mock_segmenter
    return factory


@pytest.fixture
def test_image() -> Image.Image:
    test_image_path = Path("tests/data/abema_water.png")
    return Image.open(test_image_path)
