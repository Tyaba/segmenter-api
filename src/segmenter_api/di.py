from typing import Self, TypeVar

from injector import Binder, Injector, Module, provider, singleton

from segmenter_api.domain.factory.segmenter_factory import segmenterFactoryInterface
from segmenter_api.infra.factory.segmenter_factory import segmenterFactory

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

    def resolve(self, cls: type[T]) -> T:
        return self.injector.get(cls)


class FactoryModule(Module):
    @provider
    @singleton
    def provide_segmenter_factory(self) -> segmenterFactoryInterface:
        return segmenterFactory()
