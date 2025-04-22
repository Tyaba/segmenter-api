import torch
from transformers.models.auto.modeling_auto import AutoModelForZeroShotObjectDetection
from transformers.models.auto.processing_auto import AutoProcessor

from segmenter_api.domain.service.detector import (
    Detector,
    Text2BboxInput,
    Text2BboxOutput,
)


class GroundingDinoDetector(Detector):
    def __init__(self):
        model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )

    def text2bbox(self, text2bbox_input: Text2BboxInput) -> Text2BboxOutput:
        inputs = self.processor(
            images=text2bbox_input.image,
            text=text2bbox_input.texts,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[text2bbox_input.image.size[::-1]],
        )

        result = results[0]
        labels = result["labels"]
        bboxes = result["boxes"].tolist()
        return Text2BboxOutput(
            labels=labels,
            bboxes=bboxes,
        )
