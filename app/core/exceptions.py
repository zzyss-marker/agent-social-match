from __future__ import annotations


class AppException(Exception):
    """Base exception for application-layer errors."""

    def __init__(self, message: str, status_code: int = 400, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code or self.__class__.__name__.lower()


class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class ConflictException(AppException):
    def __init__(self, message: str = "Resource already exists") -> None:
        super().__init__(message, status_code=409)


class ValidationException(AppException):
    def __init__(self, message: str = "Validation failed") -> None:
        super().__init__(message, status_code=422)
