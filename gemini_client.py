"""
modules/llm/gemini_client.py
============================
Gemini implementation of `StructuredLLMClient`.

Key design decision: **constrained decoding**.
Gemini supports `response_mime_type="application/json"` together with a
`response_schema`. Passing the Pydantic model as the schema forces the
model to emit conforming JSON at decode time, which removes the entire
class of "model returned prose / markdown fences / a wrong key" failures
that would otherwise each cost a retry call.

Supports both official SDKs:
  * `google-genai`      (new, preferred - accepts Pydantic classes directly)
  * `google-generativeai` (legacy - accepts a JSON-schema dict)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from modules.llm.base import LLMError, LLMResponse, LLMUsage
from modules.llm.cache import ResponseCache
from modules.logger import get_logger

logger = get_logger("llm.gemini")


class GeminiClient:
    """Structured-output Gemini client with transport retries and caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        max_retries: int = 2,
        timeout_s: int = 120,
        cache: Optional[ResponseCache] = None,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set - the LLM stage cannot run.")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.cache = cache

        self._sdk: str = ""       # "genai" | "generativeai"
        self._client: Any = None
        self.call_count: int = 0  # observability: real network calls made

    # ------------------------------------------------------------------ #
    # SDK bootstrap
    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> None:
        if self._client is not None:
            return

        # Preferred: new unified SDK
        try:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=self.api_key)
            self._sdk = "genai"
            logger.info("Using google-genai SDK | model=%s", self.model)
            return
        except ImportError:
            pass

        # Fallback: legacy SDK
        try:
            import google.generativeai as genai_legacy  # type: ignore

            genai_legacy.configure(api_key=self.api_key)
            self._client = genai_legacy
            self._sdk = "generativeai"
            logger.info("Using google-generativeai SDK (legacy) | model=%s", self.model)
            return
        except ImportError as exc:
            raise LLMError(
                "No Gemini SDK found. Install one of:\n"
                "  pip install google-genai          (recommended)\n"
                "  pip install google-generativeai   (legacy)"
            ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> LLMResponse:
        """
        Single schema-constrained generation.

        Checks the cache first; only on a miss is a network call made.
        """
        schema_json = json.dumps(response_model.model_json_schema(), sort_keys=True)

        cache_key = ""
        if self.cache is not None:
            cache_key = ResponseCache.build_key(
                self.model, system_instruction, user_prompt, schema_json
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return LLMResponse(
                    raw_text=cached.get("raw_text", ""),
                    parsed=cached.get("parsed"),
                    usage=LLMUsage(**cached.get("usage", {})),
                    model=self.model,
                    from_cache=True,
                )

        self._ensure_client()
        response = self._call_with_retries(system_instruction, user_prompt, response_model)

        if self.cache is not None and cache_key:
            self.cache.set(
                cache_key,
                {
                    "raw_text": response.raw_text,
                    "parsed": response.parsed,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                },
            )
        return response

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def _call_with_retries(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> LLMResponse:
        """Retry transport failures only - never schema failures."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                logger.info(
                    "Gemini call %d/%d | model=%s | prompt_chars=%d",
                    attempt,
                    self.max_retries + 1,
                    self.model,
                    len(user_prompt),
                )
                self.call_count += 1
                if self._sdk == "genai":
                    return self._call_genai(system_instruction, user_prompt, response_model)
                return self._call_legacy(system_instruction, user_prompt, response_model)
            except Exception as exc:  # network / quota / transient
                last_error = exc
                logger.warning("Gemini call failed (attempt %d): %s", attempt, exc)
                if attempt <= self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    time.sleep(backoff)

        raise LLMError(f"Gemini call failed after {self.max_retries + 1} attempts: {last_error}")

    def _call_genai(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> LLMResponse:
        """
        New google-genai SDK.

        We request JSON mode (response_mime_type) but do NOT hand the SDK a
        response_schema. Reason: the SDK re-serializes any schema (Pydantic
        class or dict) into its proto Schema form and, on several versions,
        injects `additionalProperties`/`additional_properties`, which the
        API then rejects with 400 INVALID_ARGUMENT. Instead the schema is
        described in the prompt and enforced afterwards by Pydantic
        validation - which we do anyway. This is version-proof.
        """
        from google.genai import types  # type: ignore

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        result = self._client.models.generate_content(
            model=self.model,
            contents=self._augment_prompt_with_schema(user_prompt, response_model),
            config=config,
        )

        raw_text = getattr(result, "text", "") or ""
        parsed = self._coerce_parsed(getattr(result, "parsed", None), raw_text)
        return LLMResponse(
            raw_text=raw_text,
            parsed=parsed,
            usage=self._usage_from(getattr(result, "usage_metadata", None)),
            model=self.model,
        )

    @staticmethod
    def _augment_prompt_with_schema(user_prompt: str, response_model: Type[BaseModel]) -> str:
        """
        Append a compact JSON-shape description to the prompt.

        Uses the cleaned schema (refs inlined, noise stripped) so the model
        sees field names, types, and nesting without the keywords that
        confuse it. JSON mode guarantees the output parses; this guarantees
        it has the right shape; Pydantic guarantees it validates.
        """
        import json as _json

        cleaned = _to_gemini_schema(response_model.model_json_schema())
        return (
            f"{user_prompt}\n\n"
            "Return a single JSON object conforming EXACTLY to this schema "
            "(field names, types, and nesting). Use null for any field not "
            "present in the document. Do not add fields not in the schema.\n"
            "----- JSON SCHEMA -----\n"
            f"{_json.dumps(cleaned, indent=2)}\n"
            "----- END SCHEMA -----"
        )

    def _call_legacy(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> LLMResponse:
        """
        Legacy google-generativeai SDK.

        Same strategy as the new SDK: JSON mime type, schema described in
        the prompt, enforcement by Pydantic. Avoids response_schema for the
        same additionalProperties reason.
        """
        model = self._client.GenerativeModel(
            model_name=self.model,
            system_instruction=system_instruction,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        result = model.generate_content(
            self._augment_prompt_with_schema(user_prompt, response_model)
        )
        raw_text = getattr(result, "text", "") or ""
        return LLMResponse(
            raw_text=raw_text,
            parsed=self._coerce_parsed(None, raw_text),
            usage=self._usage_from(getattr(result, "usage_metadata", None)),
            model=self.model,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_parsed(parsed: Any, raw_text: str) -> Optional[Dict[str, Any]]:
        """Normalize the SDK's parsed payload into a plain dict."""
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        if not raw_text:
            return None
        text = raw_text.strip()
        # Constrained decoding makes fences unlikely, but stay defensive.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        # Fast path: the whole thing is JSON.
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            pass
        # Fallback: extract the outermost {...} object from surrounding prose.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                loaded = json.loads(text[start : end + 1])
                return loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                pass
        logger.error("Gemini response was not valid JSON (%d chars).", len(raw_text))
        return None

    @staticmethod
    def _usage_from(usage_metadata: Any) -> LLMUsage:
        if usage_metadata is None:
            return LLMUsage()
        return LLMUsage(
            prompt_tokens=int(getattr(usage_metadata, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage_metadata, "candidates_token_count", 0) or 0),
        )


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce a Pydantic JSON schema to the subset Gemini structured output
    accepts (this subset is the same for both the new google-genai SDK and
    the legacy google-generativeai SDK).

    Transformations:
      * inline $defs / $ref
      * Optional[X] (anyOf: [X, null]) -> X with nullable=True
      * drop keywords the endpoint rejects (title, default,
        additionalProperties, description, and validation constraints such
        as minimum / minLength / pattern that Gemini does not support)
      * preserve type, properties, items, enum, and required
    """
    defs = schema.get("$defs", {})

    # Keywords Gemini's response_schema does NOT accept and that would
    # trigger "invalid JSON payload received".
    _DROP = {
        "title", "default", "additionalProperties", "$defs", "description",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "pattern", "minItems", "maxItems",
        "const", "examples", "$schema", "discriminator",
    }

    def resolve(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref_name = str(node["$ref"]).rsplit("/", 1)[-1]
            return resolve(defs.get(ref_name, {}))

        # Optional[X] arrives as anyOf: [X, null]. Collapse to the concrete
        # type but mark it nullable so the model may legitimately return null.
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            has_null = any(o.get("type") == "null" for o in node["anyOf"])
            if options:
                resolved = resolve(options[0])
                if has_null and isinstance(resolved, dict):
                    resolved["nullable"] = True
                return resolved
            return {"type": "string", "nullable": True}

        cleaned: Dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROP:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {k: resolve(v) for k, v in value.items()}
            elif key == "items":
                cleaned[key] = resolve(value)
            elif key == "required" and isinstance(value, list):
                cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned

    return resolve({k: v for k, v in schema.items() if k != "$defs"})
