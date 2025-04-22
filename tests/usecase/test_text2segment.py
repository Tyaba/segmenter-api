from PIL import Image

from segmenter_api.domain.factory.detector_factory import (
    DetectorFactoryInterface,
    DetectorType,
)
from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterFactoryInterface,
    SegmenterType,
)
from segmenter_api.domain.model.text2segment import Text2SegmentInput
from segmenter_api.usecase.service.text2segment import Text2SegmentUsecase


def test_text2segment(
    mock_detector_factory: DetectorFactoryInterface,
    mock_segmenter_factory: SegmenterFactoryInterface,
):
    # テストデータの準備
    input_image = Image.new("RGB", (100, 100))
    texts = ["test object"]

    # ユースケースのインスタンス化
    usecase = Text2SegmentUsecase(
        segmenter_factory=mock_segmenter_factory, detector_factory=mock_detector_factory
    )

    # テスト実行
    input_data = Text2SegmentInput(
        texts=texts,
        image=input_image,
        detector_type=DetectorType.FLORENCE2,
        segmenter_type=SegmenterType.SAM2,
    )

    output = usecase.text2segment(input_data)

    # アサーション
    assert len(output.masks) == 1
    assert output.labels == ["test object"]
    assert output.text2bbox_output.bboxes == [(0, 0, 50, 50)]
    assert len(output.bbox2segment_output.masks) == 1

    # モックの呼び出し確認
    mock_detector_factory.create.assert_called_once_with(DetectorType.FLORENCE2)
    mock_segmenter_factory.create.assert_called_once_with(SegmenterType.SAM2)
    mock_detector_factory.create.return_value.text2bbox.assert_called_once()
    mock_segmenter_factory.create.return_value.bbox2segment.assert_called_once()
