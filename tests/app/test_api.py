import os

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from segmenter_api.app.server import api
from segmenter_api.domain.factory.detector_factory import DetectorType
from segmenter_api.domain.factory.segmenter_factory import SegmenterType
from segmenter_api.usecase.ui.foreground_segment import ForegroundSegmentRequest
from segmenter_api.usecase.ui.text2segment import Text2SegmentRequest
from segmenter_api.utils.google import get_authorized_headers
from segmenter_api.utils.image import pil2base64


@pytest.fixture
def client():
    return TestClient(api)


@pytest.fixture
def cloud_run_url():
    return os.environ["CLOUD_RUN_URL"]


def create_text2segment_request(
    texts: list[str],
    image: Image.Image,
    detector_type: DetectorType = DetectorType.FLORENCE2_BASE,
    segmenter_type: SegmenterType = SegmenterType.SAM2,
) -> Text2SegmentRequest:
    return Text2SegmentRequest(
        texts=texts,
        image=pil2base64(image),
        detector_type=detector_type,
        segmenter_type=segmenter_type,
    )


def create_foreground_segment_request(
    image: Image.Image,
    segmenter_type: SegmenterType = SegmenterType.BIREFNET,
) -> ForegroundSegmentRequest:
    return ForegroundSegmentRequest(
        image=pil2base64(image),
        segmenter_type=segmenter_type,
    )


def verify_text2segment_response(response_data: dict, expected_label_count: int = 2):
    assert "labels" in response_data
    assert "masks" in response_data
    assert len(response_data["labels"]) == expected_label_count
    assert len(response_data["masks"]) == expected_label_count


def verify_foreground_segment_response(response_data: dict):
    assert "mask" in response_data


def call_text2segment_endpoint(
    client: TestClient | str,
    request: Text2SegmentRequest,
    timeout: int = 300,
    headers: dict | None = None,
) -> dict:
    if isinstance(client, TestClient):
        response = client.post(
            "/text2segment",
            json=request.model_dump(),
            headers=headers,
        )
    else:
        response = requests.post(
            f"{client}/text2segment",
            json=request.model_dump(),
            timeout=timeout,
            headers=headers,
        )
    assert response.status_code == 200
    return response.json()


def call_foreground_segment_endpoint(
    client: TestClient | str,
    request: ForegroundSegmentRequest,
    timeout: int = 300,
    headers: dict | None = None,
) -> dict:
    if isinstance(client, TestClient):
        response = client.post(
            "/foreground_segment",
            json=request.model_dump(),
            headers=headers,
        )
    else:
        response = requests.post(
            f"{client}/foreground_segment",
            json=request.model_dump(),
            timeout=timeout,
            headers=headers,
        )
    assert response.status_code == 200
    return response.json()


@pytest.mark.cuda
def test_text2segment_endpoint(
    client,
    test_image: Image.Image,
):
    request = create_text2segment_request(
        texts=["plastic bottle", "green character"],
        image=test_image,
    )
    response_data = call_text2segment_endpoint(
        client=client,
        request=request,
    )
    verify_text2segment_response(response_data)


@pytest.mark.cuda
def test_foreground_segment_endpoint(
    client,
    test_image: Image.Image,
):
    request = create_foreground_segment_request(image=test_image)
    response_data = call_foreground_segment_endpoint(
        client=client,
        request=request,
    )
    verify_foreground_segment_response(response_data)


@pytest.mark.cloudrun
def test_cloud_run_text2segment_endpoint(
    cloud_run_url,
    test_image: Image.Image,
):
    """Cloud Runのエンドポイントへの疎通確認テスト"""
    request = create_text2segment_request(
        texts=["plastic bottle", "green character"],
        image=test_image,
    )
    response_data = call_text2segment_endpoint(
        client=cloud_run_url,
        request=request,
        headers=get_authorized_headers(cloud_run_url),
    )
    verify_text2segment_response(response_data)


@pytest.mark.cloudrun
def test_cloud_run_foreground_segment_endpoint(
    cloud_run_url,
    test_image: Image.Image,
):
    """Cloud Runのエンドポイントへの疎通確認テスト"""
    request = create_foreground_segment_request(image=test_image)
    response_data = call_foreground_segment_endpoint(
        client=cloud_run_url,
        request=request,
        headers=get_authorized_headers(cloud_run_url),
    )
    verify_foreground_segment_response(response_data)
