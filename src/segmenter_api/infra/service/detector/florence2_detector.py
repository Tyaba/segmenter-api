import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.processing_auto import AutoProcessor

from segmenter_api.domain.service.detector import (
    Detector,
    Text2BboxInput,
    Text2BboxOutput,
)


class Florence2Detector(Detector):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
        self.task_prompt = "<OPEN_VOCABULARY_DETECTION>"

    def text2bbox(self, text2bbox_input: Text2BboxInput) -> Text2BboxOutput:
        prompts = [f"{self.task_prompt}{text}" for text in text2bbox_input.texts]
        # 前処理
        encoded = self.processor(
            text=prompts,
            images=[text2bbox_input.image] * len(prompts),
            return_tensors="pt",
        ).to(self.device, self.torch_dtype)

        # 本処理
        generated_ids = self.model.generate(
            input_ids=encoded["input_ids"],
            pixel_values=encoded["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False,
        )

        # decode
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )

        # 後処理
        parsed_answers = [
            self.processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(text2bbox_input.image.width, text2bbox_input.image.height),
            )
            for generated_text in generated_texts
        ]

        bboxes_dict: dict[str, list[float]] = {
            parsed_answer["<OPEN_VOCABULARY_DETECTION>"]["bboxes_labels"][
                0
            ]: parsed_answer["<OPEN_VOCABULARY_DETECTION>"]["bboxes"][0]
            for parsed_answer in parsed_answers
        }
        bboxes = [bboxes_dict[text] for text in text2bbox_input.texts]

        return Text2BboxOutput(bboxes=bboxes)
