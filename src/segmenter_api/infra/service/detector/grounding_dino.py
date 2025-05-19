import torch
from transformers.models.auto.modeling_auto import AutoModelForZeroShotObjectDetection
from transformers.models.auto.processing_auto import AutoProcessor

from segmenter_api.domain.service.detector import (
    Detector,
    DetectorInput,
    DetectorOutput,
)


class GroundingDinoDetector(Detector):
    def __init__(self):
        model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )

    def detect(self, detector_input: DetectorInput) -> DetectorOutput:
        inputs = self.processor(
            images=detector_input.image,
            text=detector_input.texts,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[detector_input.image.size[::-1]],
        )

        result = results[0]
        labels = result["labels"]
        bboxes = result["boxes"].tolist()
        return DetectorOutput(
            labels=labels,
            bboxes=bboxes,
        )
