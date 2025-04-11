from segmenter_api.domain.service.segmenter import (
    Bbox2SegmentInput,
    Bbox2SegmentOutput,
    Segmenter,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np
from PIL import Image
from segmenter_api.utils.time import stop_watch
class SAM2(Segmenter):
    @stop_watch
    def __init__(self):
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

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
