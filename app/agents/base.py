import json

from ..llm import GROUNDING_RULES, OpenRouterClient


class Agent:
    name = "base"

    def __init__(self, llm: OpenRouterClient):
        self.llm = llm

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    def _ctx_json(self, ctx: dict, keys: list[str]) -> str:
        payload = {k: ctx.get(k) for k in keys if k in ctx}
        return json.dumps(payload, ensure_ascii=False, default=str)

    def enabled(self) -> bool:
        return self.llm.available()

    async def run(self, ctx: dict) -> dict:
        user = self._ctx_json(ctx, self.ctx_keys)
        result = await self.llm.json_chat(self.system_prompt, user)
        return result
