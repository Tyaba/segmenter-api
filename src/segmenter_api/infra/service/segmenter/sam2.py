from segmenter_api.domain.service.segmenter import (
    Bbox2SegmentInput,
    Bbox2SegmentOutput,
    Segmenter,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import torch
import numpy as np
from PIL import Image
from segmenter_api.utils.time import stop_watch
from segmenter_api.settings import get_settings

settings = get_settings()

class SAM2(Segmenter):
    @stop_watch
    def __init__(self):
        checkpoint = str(settings.sam2_model_path / "sam2.1_hiera_large.pt")
        model_cfg = str(settings.sam2_model_path / "sam2.1_hiera_l.yaml")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    @stop_watch
    def bbox2segment(self, bbox2segment_input: Bbox2SegmentInput) -> Bbox2SegmentOutput:
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
