import torch
from dreamsim import dreamsim

from segmenter_api.domain.service.segmenter import (
    VectorizeInput,
    VectorizeOutput,
    segmenter,
)
from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)


class DreamSim(segmenter):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = dreamsim(pretrained=True, device=self.device)

    def vectorize(self, vectorize_input: VectorizeInput) -> VectorizeOutput:
        self.validate_input(vectorize_input)

        _image_tensors = [self.preprocess(image) for image in vectorize_input.images]
        images_tensor = torch.cat(_image_tensors, dim=0)
        embeddings = self.model.embed(images_tensor)
        vectors = embeddings.cpu().numpy().tolist()
        return VectorizeOutput(vectors=vectors)

    def validate_input(self, vectorize_input: VectorizeInput) -> None:
        if vectorize_input.images is None:
            error_msg = "images is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if vectorize_input.texts is not None:
            error_msg = f"texts should be empty: {vectorize_input.texts}"
            logger.error(error_msg)
            raise ValueError(error_msg)
