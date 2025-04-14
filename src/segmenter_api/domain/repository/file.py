from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path


class FileRepositoryInterface(ABC):
    @abstractmethod
    def download(
        self, source_paths: list[PathLike], destination_paths: list[Path]
    ) -> None:
        pass

    @abstractmethod
    def upload(
        self, source_paths: list[Path], destination_paths: list[PathLike]
    ) -> None:
        pass

    @abstractmethod
    def download_to_dir(
        self, source_paths: list[PathLike], destination_dir: Path
    ) -> list[Path]:
        pass

    @abstractmethod
    def upload_to_dir(
        self,
        source_paths: list[Path],
        destination_dir: PathLike,
        overwrite: bool = False,
    ) -> list[PathLike]:
        pass
