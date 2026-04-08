from typing import Annotated

from fastapi import Header, Request
from secrets import compare_digest

from .errors import ServiceError


def require_bearer_token(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    settings = request.app.state.settings

    if not authorization or not authorization.startswith("Bearer "):
        raise ServiceError("UNAUTHORIZED", "Missing or invalid bearer token.", 401)

    token = authorization.removeprefix("Bearer ").strip()
    if not token or not compare_digest(token, settings.TRANSCRIPTION_SERVICE_TOKEN):
        raise ServiceError("UNAUTHORIZED", "Missing or invalid bearer token.", 401)
