"""RFC 7807 (problem+json) error handling for the API.

Every error response is a ``application/problem+json`` body with at least
``type``/``title``/``status``/``detail`` so the frontend (and generated TS
client) sees one consistent error shape (web-app-plan §6).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_MEDIA_TYPE = "application/problem+json"


class ProblemException(Exception):
    """Raise from route/service code to emit a problem+json response.

    ``type`` defaults to ``about:blank`` per RFC 7807 when no problem-type URI
    is defined. ``extra`` merges additional members into the body.
    """

    def __init__(
        self,
        status: int,
        title: str,
        detail: str,
        *,
        type_: str = "about:blank",
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail
        self.type = type_
        self.extra = extra or {}


def problem_response(
    status: int,
    title: str,
    detail: str,
    *,
    type_: str = "about:blank",
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": type_,
        "title": title,
        "status": status,
        "detail": detail,
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status, content=body, media_type=PROBLEM_MEDIA_TYPE)


async def _problem_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProblemException)
    return problem_response(
        exc.status,
        exc.title,
        exc.detail,
        type_=exc.type,
        extra=exc.extra,
    )


async def _http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return problem_response(exc.status_code, _title_for_status(exc.status_code), detail)


async def _validation_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return problem_response(
        422,
        "Unprocessable Entity",
        "Request validation failed.",
        extra={"errors": jsonable_encoder(exc.errors())},
    )


def _title_for_status(status: int) -> str:
    return {
        400: "Bad Request",
        404: "Not Found",
        409: "Conflict",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }.get(status, "Error")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProblemException, _problem_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
