import traceback

import torch
from injector import inject
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.processing_auto import AutoProcessor

from segmenter_api.domain.repository.file import FileRepositoryInterface
from segmenter_api.domain.service.detector import (
    Detector,
    Text2BboxInput,
    Text2BboxOutput,
)
from segmenter_api.settings import get_settings
from segmenter_api.utils.logger import get_logger
from segmenter_api.utils.time import stop_watch

logger = get_logger(__name__)

settings = get_settings()


class Florence2Detector(Detector):
    @stop_watch
    @inject
    def __init__(self, file_repository: FileRepositoryInterface):
        self.file_repository = file_repository
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model, self.processor = self._load_model()
        self.task_prompt = "<OPEN_VOCABULARY_DETECTION>"

    def _load_model(self) -> tuple[AutoModelForCausalLM, AutoProcessor]:
        def _load_model_from_path(
            path: str,
        ) -> tuple[AutoModelForCausalLM, AutoProcessor]:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device)
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            return model, processor

        local_model_path = settings.florence2_model_path
        try:
            # download local model from repository
            self.file_repository.download_to_dir(
                source_paths=[local_model_path],
                destination_dir=local_model_path,
            )
            # load local model
            return _load_model_from_path(str(local_model_path))
        except Exception as e:
            logger.warning(f"ローカルモデルのロードに失敗しました: {e}")
            logger.warning(traceback.format_exc())
            logger.warning("公開モデルのロードを試みます")
            return _load_model_from_path("microsoft/Florence-2-base")

    @stop_watch
    def text2bbox(self, text2bbox_input: Text2BboxInput) -> Text2BboxOutput:
        # 前処理
        image = text2bbox_input.image.convert("RGB")
        labels: list[str] = []
        bboxes: list[tuple[float, float, float, float]] = []
        for text in text2bbox_input.texts:
            _bboxes = self._text2bbox(text=text, image=image)
            labels += [text] * len(_bboxes)
            bboxes += _bboxes
        return Text2BboxOutput(labels=labels, bboxes=bboxes)

    def _text2bbox(
        self, text: str, image: Image.Image
    ) -> list[tuple[float, float, float, float]]:
        prompt = f"{self.task_prompt}{text}"
        encoded = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, self.torch_dtype)

        # 本処理
        generated_ids = self.model.generate(
            input_ids=encoded["input_ids"],
            pixel_values=encoded["pixel_values"],
            # max_new_tokens=4096,
            max_new_tokens=1024,
            num_beams=3,
            # do_sample=False,
        )

        # decode
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # 後処理
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=image.size,
        )
        bboxes = parsed_answer["<OPEN_VOCABULARY_DETECTION>"]["bboxes"]
        bboxes_tuple = [tuple(bbox) for bbox in bboxes]
        return bboxes_tuple
