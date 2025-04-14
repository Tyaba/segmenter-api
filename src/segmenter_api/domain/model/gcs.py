from typing import Self

from pydantic import BaseModel, field_validator


class GCSPath(BaseModel):
    bucket: str
    blob_path: str

    @classmethod
    def from_path(cls, path: str) -> Self:
        """
        gs://bucket-name/path/to/file 形式の文字列からGcsPathを生成
        """
        if not path.startswith("gs://"):
            raise ValueError("GCS path must start with gs://")

        path_without_prefix = path[5:]  # Remove "gs://"
        bucket, *rest = path_without_prefix.split("/", 1)
        blob_path = rest[0] if rest else ""
        return cls(bucket=bucket, blob_path=blob_path)

    @field_validator("blob_path")
    @classmethod
    def validate_blob_path(cls, v: str) -> str:
        if v.startswith("/"):
            v = v[1:]
        return v

    def __str__(self) -> str:
        return f"gs://{self.bucket}/{self.blob_path}"

    model_config = {"frozen": True}
