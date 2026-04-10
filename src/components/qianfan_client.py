import json
from typing import Optional, List, Dict, Any

import requests


class QianfanAPIError(Exception):
    """Raised when Qianfan API returns an error or an unexpected response."""
    pass


class QianfanClient:
    """
    Minimal Qianfan chat client for text generation.

    Current design goal:
    - Keep it simple so it can plug into the existing repo quickly
    - Return plain assistant text
    - Support both system + user messages
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen2.5-7B-Instruct",
        base_url: str = "https://qianfan.baidubce.com/v2/chat/completions",
        timeout: int = 120,
        default_system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty.")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.default_system_prompt = default_system_prompt

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self.base_url,
            headers=self._build_headers(),
            data=json.dumps(payload, ensure_ascii=False),
            timeout=self.timeout,
        )

        # HTTP-level error
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise QianfanAPIError(
                f"Qianfan HTTP error: {response.status_code}, body={response.text}"
            ) from e

        # JSON-level parse
        try:
            data = response.json()
        except Exception as e:
            raise QianfanAPIError(
                f"Qianfan returned non-JSON response: {response.text}"
            ) from e

        # API-level error
        if isinstance(data, dict) and "error" in data:
            raise QianfanAPIError(f"Qianfan API error: {data}")

        if isinstance(data, dict) and "code" in data and "message" in data and "choices" not in data:
            raise QianfanAPIError(
                f"Qianfan API error: code={data.get('code')}, message={data.get('message')}, full={data}"
            )

        return data

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Direct chat call with OpenAI-like messages format, e.g.
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        if not messages:
            raise ValueError("messages must not be empty.")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens

        data = self._post(payload)

        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise QianfanAPIError(
                f"Unexpected Qianfan response format: {data}"
            ) from e

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Convenience wrapper for single-turn generation.
        """
        if not user_prompt:
            raise ValueError("user_prompt must not be empty.")

        final_system_prompt = system_prompt or self.default_system_prompt

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )