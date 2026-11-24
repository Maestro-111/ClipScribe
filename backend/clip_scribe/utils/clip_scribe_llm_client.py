"""Factory for every OpenAI-compatible LLM client in the project.

Scene analysis (``openai.OpenAI``), the parser/chat models (``ChatOpenAI``) and
the taxonomy agents (``openai-agents``) all build their client here, so pointing
ClipScribe at an OpenAI-compatible endpoint such as Azure AI Foundry is
configured in one place, from the environment:

* ``CLIPSCRIBE_LLM_BASE_URL``: endpoint, e.g. ``https://<resource>.openai.azure.com/openai/v1/``.
* ``CLIPSCRIBE_LLM_API_KEY``: key for that endpoint.

Either variable left unset falls through to the SDK default (``OPENAI_BASE_URL`` /
``OPENAI_API_KEY``), so plain OpenAI keeps working. The ``model`` passed by
callers is the deployment name when the endpoint is Azure.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from openai import OpenAI

logger = logging.getLogger("clip_scribe")

# Fallback deployment name when ``clip_scribe.yaml`` omits a model.
DEFAULT_LLM_MODEL = "gpt-6-luna"


class _ConnectionKwargs(TypedDict, total=False):
    base_url: str
    api_key: str


def _connection_kwargs() -> _ConnectionKwargs:
    """Endpoint/key overrides from the environment; only the ones that are set."""
    kwargs: _ConnectionKwargs = {}
    base_url = os.environ.get("CLIPSCRIBE_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("CLIPSCRIBE_LLM_API_KEY", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def make_openai_client() -> OpenAI:
    """Sync client for direct chat-completions calls (scene description)."""
    from openai import OpenAI

    return OpenAI(**_connection_kwargs())


def make_chat_model(model: str, **kwargs: Any) -> ChatOpenAI:
    """LangChain chat model for the parser agents and advisory chat.

    ``kwargs`` are passed through to ``ChatOpenAI`` (temperature, timeout, ...);
    ``None`` values are dropped so an unset option (e.g. ``temperature`` for
    models that only accept their default) is simply not sent.
    """
    from langchain_openai import ChatOpenAI

    options = {k: v for k, v in kwargs.items() if v is not None}
    return ChatOpenAI(model=model, **_connection_kwargs(), **options)  # type: ignore[arg-type]


def configure_agents_sdk() -> None:
    """Point the ``openai-agents`` default client at the configured endpoint.

    ``Agent``/``Runner`` build their own client from ``OPENAI_API_KEY`` unless a
    default is registered, so this must run before the first ``Runner`` call.
    A no-op when no override is set. For a custom endpoint it also switches to
    chat completions (the API every Foundry deployment supports) and disables
    tracing, which would otherwise upload to platform.openai.com and fail
    without a real OpenAI key.
    """
    connection = _connection_kwargs()
    if not connection:
        return

    from agents import (
        set_default_openai_api,
        set_default_openai_client,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_default_openai_client(AsyncOpenAI(**connection), use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    logger.info(
        "ClipScribe LLM: openai-agents routed to custom endpoint "
        f"{connection.get('base_url', '<OPENAI_BASE_URL>')}"
    )
