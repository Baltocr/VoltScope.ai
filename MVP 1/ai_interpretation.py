"""OpenAI backend for VoltScope AI interpretation.

This module only accepts the compact interpretation prompt built from computed
VoltScope metrics. It never reads uploaded files or raw trace arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


DEFAULT_AI_MODEL = "gpt-4.1-mini"
NOT_CONFIGURED_MESSAGE = (
    "AI interpretation is not configured. Add OPENAI_API_KEY to Streamlit secrets or your environment."
)


@dataclass
class OpenAIInterpretationResult:
    content: str
    error: str
    configured: bool
    model: str


def _read_secret_value(secrets: Any, key: str) -> str:
    if secrets is None:
        return ""
    try:
        value = secrets[key]
    except Exception:
        return ""
    return str(value or "").strip()


def resolve_openai_api_key(
    *,
    streamlit_secrets: Any = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Read the OpenAI API key from Streamlit secrets first, then env."""
    secret_value = _read_secret_value(streamlit_secrets, "OPENAI_API_KEY")
    if secret_value:
        return secret_value

    env = environ if environ is not None else os.environ
    return str(env.get("OPENAI_API_KEY", "") or "").strip()


def resolve_openai_model(
    *,
    streamlit_secrets: Any = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    secret_model = _read_secret_value(streamlit_secrets, "VOLTSCOPE_AI_MODEL")
    if secret_model:
        return secret_model

    env = environ if environ is not None else os.environ
    return str(env.get("VOLTSCOPE_AI_MODEL", "") or "").strip() or DEFAULT_AI_MODEL


def request_openai_interpretation(
    prompt: str,
    *,
    streamlit_secrets: Any = None,
    environ: Optional[Mapping[str, str]] = None,
    model: Optional[str] = None,
    timeout_s: int = 45,
) -> OpenAIInterpretationResult:
    api_key = resolve_openai_api_key(streamlit_secrets=streamlit_secrets, environ=environ)
    selected_model = model or resolve_openai_model(streamlit_secrets=streamlit_secrets, environ=environ)
    if not api_key:
        return OpenAIInterpretationResult(
            content="",
            error=NOT_CONFIGURED_MESSAGE,
            configured=False,
            model=selected_model,
        )

    try:
        from openai import OpenAI, OpenAIError
    except ImportError:
        return OpenAIInterpretationResult(
            content="",
            error="OpenAI Python SDK is not installed. Install requirements.txt, then restart Streamlit.",
            configured=True,
            model=selected_model,
        )

    try:
        client = OpenAI(api_key=api_key, timeout=timeout_s)
        response = client.chat.completions.create(
            model=selected_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write careful, concise electrochemistry interpretations from supplied "
                        "VoltScope metrics only. Return a JSON object with the requested section keys."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
    except OpenAIError as exc:
        return OpenAIInterpretationResult(
            content="",
            error=f"AI interpretation request failed: {exc}",
            configured=True,
            model=selected_model,
        )
    except Exception as exc:
        return OpenAIInterpretationResult(
            content="",
            error=f"AI interpretation request failed: {exc}",
            configured=True,
            model=selected_model,
        )

    if not content:
        return OpenAIInterpretationResult(
            content="",
            error="AI interpretation request returned no content.",
            configured=True,
            model=selected_model,
        )

    return OpenAIInterpretationResult(
        content=content,
        error="",
        configured=True,
        model=selected_model,
    )
