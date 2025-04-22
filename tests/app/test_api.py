import pytest
from fastapi.testclient import TestClient
from PIL import Image

from segmenter_api.app.server import api
from segmenter_api.domain.factory.detector_factory import (
    DetectorType,
)
from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterType,
)
from segmenter_api.usecase.ui.foreground_segment import ForegroundSegmentRequest
from segmenter_api.usecase.ui.text2segment import Text2SegmentRequest
from segmenter_api.utils.image import pil2base64


@pytest.fixture
def client():
    return TestClient(api)


@pytest.mark.cuda
def test_text2segment_endpoint(
    client,
    test_image: Image.Image,
):
    # リクエストの作成
    request = Text2SegmentRequest(
        texts=["test object"],
        image=pil2base64(test_image),
        detector_type=DetectorType.FLORENCE2,
        segmenter_type=SegmenterType.SAM2,
    )

    # APIリクエストの実行
    response = client.post("/text2segment", json=request.model_dump())

    # レスポンスの検証
    assert response.status_code == 200
    data = response.json()
    assert "labels" in data
    assert "masks" in data


@pytest.mark.cuda
def test_foreground_segment_endpoint(
    client,
    test_image: Image.Image,
):
    # リクエストの作成
    request = ForegroundSegmentRequest(
        image=pil2base64(test_image), segmenter_type=SegmenterType.BIREFNET
    )

    # APIリクエストの実行
    response = client.post("/foreground_segment", json=request.model_dump())

    # レスポンスの検証
    assert response.status_code == 200
    data = response.json()
    assert "mask" in data
