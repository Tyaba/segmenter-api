from PIL import Image

from segmenter_api.domain.factory.segmenter_factory import (
    SegmenterFactoryInterface,
    SegmenterType,
)
from segmenter_api.domain.model.foreground_segment import ForegroundSegmentUsecaseInput
from segmenter_api.usecase.service.foreground_segment import ForegroundSegmentUsecase


def test_foreground_segment(mock_segmenter_factory: SegmenterFactoryInterface):
    # テストデータの準備
    input_image = Image.new("RGB", (100, 100))

    # ユースケースのインスタンス化
    usecase = ForegroundSegmentUsecase(segmenter_factory=mock_segmenter_factory)

    # テスト実行
    input_data = ForegroundSegmentUsecaseInput(
        image=input_image, segmenter_type=SegmenterType.BIREFNET
    )

    output = usecase.foreground_segment(input_data)

    # アサーション
    assert output.mask.size == (100, 100)

    # モックの呼び出し確認
    mock_segmenter_factory.create.assert_called_once_with(SegmenterType.BIREFNET)
    mock_segmenter_factory.create.return_value.foreground_segment.assert_called_once()
