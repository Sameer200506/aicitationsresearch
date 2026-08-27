import asyncio
import json
import re

import httpx

from .config import DEPRIORITY_SUBSTRINGS, settings

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

GROUNDING_RULES = """HARD RULES (must never be violated):
1. Never invent a citation, case name, or statute reference.
2. Never claim a case exists unless it appears in the provided DATABASE CONTEXT.
3. Never say a case supports a proposition unless the provided evidence text shows it.
4. Every legal claim must reference the source material you were given.
5. The original judgment text outranks any AI summary or general knowledge.
6. If something is uncertain, explicitly mark it UNCERTAIN.
If the database context does not contain sufficient authority, say so plainly."""


class OpenRouterClient:
    def __init__(self):
        self._models_cache: list[str] | None = None
        self._resolved: list[str] = []
        self.headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": settings.site_url,
            "X-Title": settings.site_name,
            "Content-Type": "application/json",
        }

    def available(self) -> bool:
        return settings.llm_enabled

    def _rank_models(self, ids: list[str]) -> list[str]:
        free = [m for m in ids if m.endswith(":free")]
        ranked: list[str] = []
        for pref in settings.model_preferences:
            matches = [m for m in free if pref.lower() in m.lower() and m not in ranked]
            if matches:
                ranked.append(matches[0])
        leftovers = [
            m for m in free
            if m not in ranked and not any(d.lower() in m.lower() for d in DEPRIORITY_SUBSTRINGS)
        ]
        ranked.extend(leftovers)
        deprioritized = [m for m in free if m not in ranked]
        ranked.extend(deprioritized)
        return ranked

    async def discover_models(self, force: bool = False) -> list[str]:
        if self._models_cache is not None and not force:
            return self._models_cache
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(f"{OPENROUTER_BASE}/models", headers=self.headers)
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", [])]
            self._models_cache = ids
            self._resolved = self._rank_models(ids)
        except Exception as e:
            if self._models_cache is None:
                raise RuntimeError(f"Could not fetch OpenRouter model list: {e}")
        return self.models()

    def models(self) -> list[str]:
        chain = []
        if settings.openrouter_model:
            chain.append(settings.openrouter_model)
        chain.extend(m for m in self._resolved if m not in chain)
        return chain or list(settings.model_preferences)

    async def chat(self, messages: list[dict], temperature: float | None = None,
                   max_tokens: int | None = None, json_mode: bool = False) -> str:
        if not self.available():
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        await self.discover_models()
        errors = []
        for model in self.models():
            payload = {
                "model": model,
                "messages": messages,
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "max_tokens": max_tokens or settings.llm_max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            for attempt in range(settings.max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                        r = await client.post(f"{OPENROUTER_BASE}/chat/completions",
                                              headers=self.headers, json=payload)
                        if r.status_code == 429:
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        r.raise_for_status()
                        data = r.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content")
                        if content:
                            return content
                        errors.append(f"{model}: empty content")
                        break
                except Exception as e:
                    msg = str(e)
                    if hasattr(e, "response") and getattr(e, "response", None) is not None:
                        try:
                            msg += f" :: {e.response.text[:300]}"
                        except Exception:
                            pass
                    errors.append(f"{model}: {msg}")
                    await asyncio.sleep(1)
        raise RuntimeError("All OpenRouter models failed. Errors:\n" + "\n".join(errors))

    @staticmethod
    def extract_json(text: str):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        # 1. Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # 2. Find outermost { ... } or [ ... ]
        # ponytail: greedy scan handles prose wrapping the JSON
        for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        # 3. Try stripping common LLM preamble like "Here is the JSON:" 
        lines = cleaned.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                remainder = "\n".join(lines[i:])
                try:
                    return json.loads(remainder)
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Model returned non-JSON output: {text[:200]}")

    async def json_chat(self, system: str, user: str, temperature: float = 0.1, max_tokens: int | None = None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            raw = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
            return self.extract_json(raw)
        except (ValueError, RuntimeError):
            raw = await self.chat(messages, temperature=0.0, max_tokens=max_tokens)
            parsed = self.extract_json(raw)
            return parsed

    async def judge_proposition(self, proposition: str, case: dict, paragraphs: list) -> dict:
        paras_block = "\n".join(
            f"[Para {num or 'n/a'}] {text}" for num, text in paragraphs
        )[:12000]
        system = (
            "You are a rigorous legal verification engine for Indian law. You compare a claimed legal "
            "proposition against stored judgment excerpts and decide whether the excerpt genuinely supports it.\n"
            + GROUNDING_RULES
        )
        user = json.dumps({
            "task": "Does the following judgment text support this proposition? Score support 0.0-1.0.",
            "proposition": proposition,
            "case_name": case.get("case_name"),
            "citation": case.get("reported_citation"),
            "judgment_excerpts": paras_block,
            "respond_with_json_schema": {
                "score": "float 0-1",
                "verdict": "'supports' | 'partially supports' | 'does not support' | 'unclear'",
                "relevant_paragraphs": [{"paragraph_number": "...", "quote": "..."}],
                "reasoning": "short justification citing only given excerpts",
            },
        }, ensure_ascii=False)
        data = await self.json_chat(system, user)
        score = max(0.0, min(1.0, float(data.get("score", 0))))
        return {
            "score": round(score, 2),
            "method": "llm",
            "verdict": data.get("verdict", "unclear"),
            "relevant_paragraphs": data.get("relevant_paragraphs", []),
            "reasoning": data.get("reasoning", ""),
        }

    def judge_proposition_sync(self, proposition: str, case: dict, paragraphs: list) -> dict:
        return asyncio.run(self.judge_proposition(proposition, case, paragraphs))


_client: OpenRouterClient | None = None


def get_llm() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client
