from http import HTTPStatus


class BaseApiError(Exception):
    status_code: HTTPStatus

    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.__class__.__name__} has occurred. Message: {self.msg}"


class BadRequestError(BaseApiError):
    status_code = HTTPStatus.BAD_REQUEST


class NotSupportedError(BaseApiError):
    status_code = HTTPStatus.BAD_REQUEST


class NotFoundError(BaseApiError):
    status_code = HTTPStatus.NOT_FOUND


class InternalServerError(BaseApiError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
