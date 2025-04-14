from typing import Self, TypeVar

from injector import Binder, Injector, Module, provider, singleton

from segmenter_api.domain.factory.detector_factory import DetectorFactoryInterface
from segmenter_api.domain.factory.segmenter_factory import SegmenterFactoryInterface
from segmenter_api.domain.repository.file import FileRepositoryInterface
from segmenter_api.infra.factory.detector_factory import DetectorFactory
from segmenter_api.infra.factory.segmenter_factory import SegmenterFactory
from segmenter_api.infra.repository.gcs import GCSRepository

T = TypeVar("T")

_di_instance = None


def resolve(cls: type[T]) -> T:
    return DI.get_instance().resolve(cls)


class DI:
    def __init__(self) -> None:
        self.injector = Injector(self._configure)

    @classmethod
    def get_instance(cls) -> Self:
        global _di_instance  # noqa: PLW0603
        if _di_instance is None:
            _di_instance = cls()
        return _di_instance

    def _configure(self, binder: Binder) -> None:
        binder.install(FactoryModule())
        binder.install(RepositoryModule())

    def resolve(self, cls: type[T]) -> T:
        return self.injector.get(cls)


class FactoryModule(Module):
    @provider
    @singleton
    def provide_segmenter_factory(self) -> SegmenterFactoryInterface:
        return SegmenterFactory()

    @provider
    @singleton
    def provide_detector_factory(self) -> DetectorFactoryInterface:
        return DetectorFactory()


class RepositoryModule(Module):
    @provider
    @singleton
    def provide_gcs_repository(self) -> FileRepositoryInterface:
        return GCSRepository()
