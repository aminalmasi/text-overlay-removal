"""text_layover, the Gemini half: which calls to make for one photo, made exactly as they were tested.

    from google import genai
    import tl_gemini
    client = genai.Client(api_key=..., http_options=tl_gemini.http_options())
    r = tl_gemini.process(client, image_bytes, "image/jpeg")
    r["eligible"], r["text"]      # production's two fields, with production's meaning
    r["erase"]                    # True -> send the photo and r["worker_reply"] to the GPU eraser

Routing (one gate call on every photo, at most one more call):

    gate P84  ->  removal 1                 -> worker P92   (text + boxes; erase on the GPU)
              ->  eligible, removal 0       -> PROD         (today's production prompt: text only)
              ->  neither                   -> done         (no text, nothing to erase)

Text-only photos (screenshots, infographics, typed bands) go to PROD, not the worker: they are never erased,
and the worker boxes every line of them (about 1,400 output tokens, roughly 6x the cost of a PROD call).

If a call fails after its retries, the photo falls back to today's behaviour: PROD for the text, no erasure.
Only when PROD fails too are `eligible`/`text` unknown: they come back as None with `error` set, never as a
made-up "no text". Check `error` (or `None`) before storing the fields.

Every call: gemini-3.5-flash-lite, prompt as system_instruction, the photo as the only content, JSON schema,
temperature 0, thinking LOW, media_resolution unset, flex tier. These are the settings of the tested runs.

The prompt texts (prompts/<name>.txt) and response schemas (schemas/<name>.json) are not published with this
repository; README.md in this folder describes what each prompt asks for and every field it returns.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import httpx
from google.genai import errors, types

HERE = Path(__file__).resolve().parent
MODEL = "gemini-3.5-flash-lite"
SERVICE_TIER = "flex"
TIMEOUT_MS = 600_000         # the tested runs used 10 minutes: flex calls can queue
NAMES = ("P84_gate", "P92_worker", "PROD_text")
PROMPTS = {n: (HERE / "prompts" / f"{n}.txt").read_text(encoding="utf-8") for n in NAMES}
SCHEMAS = {n: json.loads((HERE / "schemas" / f"{n}.json").read_text(encoding="utf-8")) for n in NAMES}
MAX_OUTPUT_TOKENS = {"P84_gate": 2000, "P92_worker": 8192, "PROD_text": 2000}
RETRYABLE = {429, 500, 502, 503, 504}
ATTEMPTS = 6                  # back-off 1, 2, 4, 8, 16 s (x0.5-1.5): about half a minute in all


def http_options():
    return types.HttpOptions(timeout=TIMEOUT_MS)


def config_for(name: str, service_tier: str | None = SERVICE_TIER) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=PROMPTS[name], temperature=0, max_output_tokens=MAX_OUTPUT_TOKENS[name],
        response_mime_type="application/json", response_schema=SCHEMAS[name],
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        service_tier=service_tier)


class CallFailed(Exception):
    pass


def call(client, name: str, image_bytes: bytes, mime_type: str, service_tier: str | None = SERVICE_TIER) -> dict:
    """One call. Returns {"reply": dict, "usage": {...}, "seconds": float}. Retries 429/5xx with back-off;
    raises CallFailed on a permanent error, a cut-off reply (MAX_TOKENS) or a reply that is not JSON."""
    last = None
    for attempt in range(ATTEMPTS):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=MODEL, contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                config=config_for(name, service_tier))
        except errors.APIError as e:
            last = e
            if getattr(e, "code", None) in RETRYABLE and attempt + 1 < ATTEMPTS:
                time.sleep(min(60.0, 2 ** attempt) * (0.5 + random.random()))
                continue
            raise CallFailed(f"{name}: API error {getattr(e, 'code', '?')}: {e}") from e
        except (httpx.TransportError, ConnectionError, TimeoutError) as e:   # timeouts, resets: the SDK does not retry
            last = e
            if attempt + 1 < ATTEMPTS:
                time.sleep(min(60.0, 2 ** attempt) * (0.5 + random.random()))
                continue
            raise CallFailed(f"{name}: network error: {e!r}") from e
        cand = (response.candidates or [None])[0]
        finish = str(getattr(cand, "finish_reason", "") or "")
        if "MAX_TOKENS" in finish:
            raise CallFailed(f"{name}: reply cut off at {MAX_OUTPUT_TOKENS[name]} tokens")
        try:
            reply = json.loads(response.text)
        except Exception as e:
            raise CallFailed(f"{name}: reply is not JSON (finish {finish})") from e
        if not isinstance(reply, dict):
            raise CallFailed(f"{name}: reply is JSON but not an object")
        u = response.usage_metadata
        usage = {"input": getattr(u, "prompt_token_count", None),
                 "output": (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)}
        return {"reply": reply, "usage": usage, "seconds": round(time.perf_counter() - t0, 3)}
    raise CallFailed(f"{name}: gave up after {ATTEMPTS} attempts: {last}")


def has_added_box(worker_reply: dict) -> bool:
    return any(isinstance(b, dict) and b.get("physically_in_scene") is False for b in worker_reply.get("boxes") or [])


def _text(reply):
    t = reply.get("text", "")
    return t if isinstance(t, str) else ""


def process(client, image_bytes: bytes, mime_type: str = "image/jpeg", service_tier: str | None = SERVICE_TIER,
            fallback_service_tier: str | None = None) -> dict:
    """Gate, then at most one more call. Returns eligible/text (production's fields), removal, erase,
    worker_reply (for the GPU eraser when erase is True), the calls made and their token usage, and `error`
    when something failed. fallback_service_tier: tier for the PROD fallback ("standard" costs 2x but queues less);
    default: the same tier."""
    calls = []
    fb_tier = fallback_service_tier or service_tier
    try:
        g = call(client, "P84_gate", image_bytes, mime_type, service_tier)
        calls.append(("P84_gate", g["usage"], g["seconds"]))
        gate = g["reply"]
    except CallFailed as e:
        return _fallback(client, image_bytes, mime_type, fb_tier, calls, f"gate failed: {e}")
    eligible, removal = gate.get("eligible") is True, 1 if gate.get("removal") == 1 else 0
    if removal == 1:
        try:
            w = call(client, "P92_worker", image_bytes, mime_type, service_tier)
            calls.append(("P92_worker", w["usage"], w["seconds"]))
        except CallFailed as e:
            return _fallback(client, image_bytes, mime_type, fb_tier, calls, f"worker failed: {e}")
        reply = w["reply"]
        # production's convention: no eligible text -> text is empty (an emoji or a sticker alone can be removal 1)
        return {"eligible": eligible, "text": _text(reply) if eligible else "", "removal": 1,
                "erase": has_added_box(reply), "worker_reply": reply, "calls": calls}
    if eligible:
        try:
            p = call(client, "PROD_text", image_bytes, mime_type, fb_tier)
            calls.append(("PROD_text", p["usage"], p["seconds"]))
        except CallFailed as e:
            return {"eligible": True, "text": None, "removal": 0, "erase": False, "worker_reply": None, "calls": calls,
                    "error": f"text call failed: {e}"}
        return {"eligible": True, "text": _text(p["reply"]), "removal": 0, "erase": False,
                "worker_reply": None, "calls": calls}
    return {"eligible": False, "text": "", "removal": 0, "erase": False, "worker_reply": None, "calls": calls}


def _fallback(client, image_bytes, mime_type, service_tier, calls, why):
    """Today's behaviour: PROD alone decides eligible and copies the text; nothing is erased. If PROD fails too,
    eligible and text are unknown (None) and `error` says why."""
    try:
        p = call(client, "PROD_text", image_bytes, mime_type, service_tier)
        calls.append(("PROD_text", p["usage"], p["seconds"]))
        return {"eligible": p["reply"].get("eligible") is True, "text": _text(p["reply"]), "removal": None,
                "erase": False, "worker_reply": None, "calls": calls, "error": why}
    except CallFailed as e:
        return {"eligible": None, "text": None, "removal": None, "erase": False, "worker_reply": None,
                "calls": calls, "error": f"{why}; fallback failed: {e}"}
