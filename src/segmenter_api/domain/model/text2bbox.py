from typing import Self

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    model_validator,
)

from segmenter_api.domain.factory.detector_factory import DetectorType


class Text2BboxInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    texts: list[str]
    image: Image.Image
    detector_type: DetectorType


class Text2BboxOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    bboxes: list[tuple[float, float, float, float]]
    labels: list[str]

    @model_validator(mode="after")
    def check_bboxes_and_labels(self) -> Self:
        if len(self.bboxes) != len(self.labels):
            error_msg = "bboxesとlabelsの長さが一致しません"
            raise ValueError(error_msg)
        return self


class Text2BboxRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    detector_type: DetectorType
    image: str
    texts: list[str]

    @field_serializer("detector_type")
    def serialize_detector_type(self, detector_type: DetectorType) -> str:
        return detector_type.value


class Text2BboxResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    labels: list[str]
    bboxes: list[list[float]]

    @model_validator(mode="after")
    def check_masks_and_labels_and_bboxes(self) -> Self:
        if len(self.bboxes) != len(self.labels):
            error_msg = "bboxesとlabelsの長さが一致しません"
            raise ValueError(error_msg)
        return self
