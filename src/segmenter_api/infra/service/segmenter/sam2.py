import traceback

import numpy as np
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from injector import inject
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from segmenter_api.domain.repository.file import FileRepositoryInterface
from segmenter_api.domain.service.segmenter import (
    Bbox2SegmentInput,
    Bbox2SegmentOutput,
    ForegroundSegmentInput,
    ForegroundSegmentOutput,
    Segmenter,
)
from segmenter_api.settings import get_settings
from segmenter_api.utils.logger import get_logger
from segmenter_api.utils.time import stop_watch

logger = get_logger(__name__)

settings = get_settings()


class SAM2(Segmenter):
    @inject
    @stop_watch
    def __init__(self, file_repository: FileRepositoryInterface):
        """Initialize SAM2 model.
        We have to configure hydra manually because of:
        https://github.com/facebookresearch/sam2/issues/573
        """
        self.file_repository = file_repository
        self._load_model()

    def _load_model(self):
        try:
            # download local model from repository
            self.file_repository.download_to_dir(
                source_paths=[settings.sam2_model_path],
                destination_dir=settings.sam2_model_path,
            )
            checkpoint = str(settings.sam2_model_path / "sam2.1_hiera_large.pt")
            GlobalHydra.instance().clear()
            initialize_config_dir(
                config_dir=str(settings.sam2_model_path.absolute()),
                version_base="1.3",
            )
            self.predictor = SAM2ImagePredictor(
                build_sam2("sam2.1_hiera_l", checkpoint)
            )
        except Exception as e:
            logger.warning(f"ローカルモデルのロードに失敗しました: {e}")
            logger.warning(traceback.format_exc())
            logger.warning("公開モデルのロードを試みます")
            self.predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-large"
            )

    @stop_watch
    def bbox2segment(self, bbox2segment_input: Bbox2SegmentInput) -> Bbox2SegmentOutput:
        if len(bbox2segment_input.bboxes) == 0:
            logger.warning("input bboxが空です")
            return Bbox2SegmentOutput(masks=[])
        mask_images: list[Image.Image] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image=bbox2segment_input.image.convert("RGB"))
            binary_masks, _, _ = self.predictor.predict(
                box=np.array(bbox2segment_input.bboxes),
                multimask_output=False,
            )
        if binary_masks.ndim == 4:
            binary_masks = binary_masks.squeeze(1)
        masks = np.uint8(binary_masks) * 255
        for mask in masks:
            mask_image = Image.fromarray(mask)
            mask_images.append(mask_image)
        return Bbox2SegmentOutput(masks=mask_images)

    def foreground_segment(
        self, foreground_segment_input: ForegroundSegmentInput
    ) -> ForegroundSegmentOutput:
        error_msg = "SAM2は前景抽出に対応していません"
        raise NotImplementedError(error_msg)
