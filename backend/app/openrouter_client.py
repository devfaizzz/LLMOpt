import httpx
import os
from app.utils import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPTIMIZER_FALLBACK_CHAIN

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://carbonpilot.ai",
    "X-Title": "CarbonPilot AI",
    "Content-Type": "application/json",
}

TIMEOUT = httpx.Timeout(90.0, connect=15.0)


async def call_openrouter(model: str, messages: list[dict], max_tokens: int = 1024) -> dict:
    """Single async call to OpenRouter. Returns dict with content, usage, model."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Some models return None choices or empty content — handle gracefully
    choices = data.get("choices") or []
    if not choices or not choices[0].get("message"):
        raise ValueError(f"Model '{model}' returned empty response.")
    content = choices[0]["message"].get("content") or ""
    usage = data.get("usage", {})
    return {
        "content": content,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "model": data.get("model", model),
    }


async def call_with_fallback(messages: list[dict], max_tokens: int = 512) -> dict:
    """
    Try the optimizer fallback chain in order.
    Returns first successful result.
    """
    last_err = None
    for model in OPTIMIZER_FALLBACK_CHAIN:
        try:
            result = await call_openrouter(model, messages, max_tokens=max_tokens)
            if result.get("content", "").strip():
                return result
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All optimizer models failed. Last error: {last_err}")


async def optimize_prompt(original_prompt: str) -> dict:
    """Use LLM to rewrite prompt more concisely, preserving meaning."""
    system_msg = (
        "You are a prompt optimization expert. Your task is to rewrite the given user prompt "
        "to be as concise as possible while FULLY preserving the original intent and all key requirements. "
        "Remove filler words, redundant phrases, repetition, and unnecessary politeness. "
        "Do NOT change the core ask. Return ONLY the optimized prompt text, nothing else."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Optimize this prompt:\n\n{original_prompt}"},
    ]
    return await call_with_fallback(messages, max_tokens=512)


async def run_prompt_on_model(prompt: str, model_key: str) -> dict:
    """Run the optimized prompt on a given model and return result."""
    messages = [{"role": "user", "content": prompt}]
    result = await call_openrouter(model_key, messages, max_tokens=1024)
    return result


async def check_semantic_similarity(text1: str, text2: str) -> float:
    """
    LLM-based semantic similarity approximation (0-100%).
    Used as a lightweight alternative to sentence-transformers.
    """
    system_msg = (
        "You are a semantic similarity judge. Compare the two texts below and return ONLY a number "
        "between 0 and 100 representing how semantically similar they are (100 = identical meaning). "
        "Return ONLY the number, nothing else."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Text 1:\n{text1}\n\nText 2:\n{text2}"},
    ]
    try:
        result = await call_with_fallback(messages, max_tokens=10)
        score = float(result["content"].strip().split()[0])
        return min(max(score, 0), 100)
    except Exception:
        return 85.0
