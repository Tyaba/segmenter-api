import torch
from PIL import Image
from torchvision import transforms
from transformers.models.auto.modeling_auto import AutoModelForImageSegmentation

from segmenter_api.domain.service.segmenter import (
    Bbox2SegmentInput,
    Bbox2SegmentOutput,
    ForegroundSegmentInput,
    ForegroundSegmentOutput,
    Segmenter,
)
from segmenter_api.utils.image import resize_image_keep_aspect


class BiRefNet(Segmenter):
    def __init__(self):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "zhengpeng7/BiRefNet", trust_remote_code=True
        )
        torch.set_float32_matmul_precision(["high", "highest"][0])
        self.model.to("cuda")
        self.model.eval()
        self.model.half()
        self.birefnet_image_size = (1024, 1024)

    def foreground_segment(
        self, foreground_segment_input: ForegroundSegmentInput
    ) -> ForegroundSegmentOutput:
        # resize
        resized_image = resize_image_keep_aspect(
            img=foreground_segment_input.image,
            long_size=max(self.birefnet_image_size),
        ).convert("RGB")
        input_image = Image.new("RGB", self.birefnet_image_size)
        input_image.paste(resized_image)

        transform_image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        input_images = transform_image(input_image).unsqueeze(0).to("cuda").half()

        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(input_image.size)
        # 元サイズに戻す
        output_mask = mask.crop(
            (0, 0, resized_image.size[0], resized_image.size[1])
        ).resize(foreground_segment_input.image.size)
        return ForegroundSegmentOutput(mask=output_mask)

    def bbox2segment(self, bbox2segment_input: Bbox2SegmentInput) -> Bbox2SegmentOutput:
        error_msg = "BiRefNetはbbox2segmentに対応していません"
        raise NotImplementedError(error_msg)
