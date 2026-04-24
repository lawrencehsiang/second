# import json
# from typing import Optional, List, Dict, Any

# import requests


# class QianfanAPIError(Exception):
#     """Raised when Qianfan API returns an error or an unexpected response."""
#     pass


# class QianfanClient:
#     """
#     Minimal Qianfan chat client for text generation.

#     Current design goals:
#     - Keep existing generate()/chat() interface unchanged (return plain text)
#     - Add generate_with_usage()/chat_with_usage() for token logging
#     - Support both system + user messages
#     """

#     def __init__(
#         self,
#         api_key: str,
#         model: str = "Qwen2.5-7B-Instruct",
#         base_url: str = "https://qianfan.baidubce.com/v2/chat/completions",
#         timeout: int = 120,
#         default_system_prompt: str = "You are a helpful assistant.",
#     ) -> None:
#         if not api_key:
#             raise ValueError("api_key must not be empty.")
#         self.api_key = api_key
#         self.model = model
#         self.base_url = base_url
#         self.timeout = timeout
#         self.default_system_prompt = default_system_prompt

#     def _build_headers(self) -> Dict[str, str]:
#         return {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.api_key}",
#         }

#     def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
#         response = requests.post(
#             self.base_url,
#             headers=self._build_headers(),
#             data=json.dumps(payload, ensure_ascii=False),
#             timeout=self.timeout,
#         )

#         # HTTP-level error
#         try:
#             response.raise_for_status()
#         except requests.HTTPError as e:
#             raise QianfanAPIError(
#                 f"Qianfan HTTP error: {response.status_code}, body={response.text}"
#             ) from e

#         # JSON-level parse
#         try:
#             data = response.json()
#         except Exception as e:
#             raise QianfanAPIError(
#                 f"Qianfan returned non-JSON response: {response.text}"
#             ) from e

#         # API-level error
#         if isinstance(data, dict) and "error" in data:
#             raise QianfanAPIError(f"Qianfan API error: {data}")

#         if (
#             isinstance(data, dict)
#             and "code" in data
#             and "message" in data
#             and "choices" not in data
#         ):
#             raise QianfanAPIError(
#                 f"Qianfan API error: code={data.get('code')}, "
#                 f"message={data.get('message')}, full={data}"
#             )

#         return data

#     def _extract_content(self, data: Dict[str, Any]) -> str:
#         try:
#             return data["choices"][0]["message"]["content"]
#         except Exception as e:
#             raise QianfanAPIError(
#                 f"Unexpected Qianfan response format: {data}"
#             ) from e

#     def _extract_usage(self, data: Dict[str, Any]) -> Dict[str, int]:
#         usage = data.get("usage", {}) if isinstance(data, dict) else {}
#         return {
#             "prompt_tokens": int(usage.get("prompt_tokens", 0)),
#             "completion_tokens": int(usage.get("completion_tokens", 0)),
#             "total_tokens": int(usage.get("total_tokens", 0)),
#         }

#     def chat(
#         self,
#         messages: List[Dict[str, str]],
#         temperature: Optional[float] = None,
#         top_p: Optional[float] = None,
#         max_output_tokens: Optional[int] = None,
#     ) -> str:
#         """
#         Direct chat call with OpenAI-like messages format.

#         Returns plain assistant text only.
#         """
#         result = self.chat_with_usage(
#             messages=messages,
#             temperature=temperature,
#             top_p=top_p,
#             max_output_tokens=max_output_tokens,
#         )
#         return result["content"]

#     def chat_with_usage(
#         self,
#         messages: List[Dict[str, str]],
#         temperature: Optional[float] = None,
#         top_p: Optional[float] = None,
#         max_output_tokens: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         """
#         Direct chat call with OpenAI-like messages format.

#         Returns:
#         {
#             "content": str,
#             "usage": {
#                 "prompt_tokens": int,
#                 "completion_tokens": int,
#                 "total_tokens": int
#             },
#             "raw_response": dict
#         }
#         """
#         if not messages:
#             raise ValueError("messages must not be empty.")

#         payload: Dict[str, Any] = {
#             "model": self.model,
#             "messages": messages,
#         }

#         if temperature is not None:
#             payload["temperature"] = temperature
#         if top_p is not None:
#             payload["top_p"] = top_p
#         if max_output_tokens is not None:
#             payload["max_output_tokens"] = max_output_tokens

#         data = self._post(payload)

#         return {
#             "content": self._extract_content(data),
#             "usage": self._extract_usage(data),
#             "raw_response": data,
#         }

#     def generate(
#         self,
#         user_prompt: str,
#         system_prompt: Optional[str] = None,
#         temperature: Optional[float] = None,
#         top_p: Optional[float] = None,
#         max_output_tokens: Optional[int] = None,
#     ) -> str:
#         """
#         Convenience wrapper for single-turn generation.

#         Returns plain assistant text only.
#         """
#         result = self.generate_with_usage(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             temperature=temperature,
#             top_p=top_p,
#             max_output_tokens=max_output_tokens,
#         )
#         return result["content"]

#     def generate_with_usage(
#         self,
#         user_prompt: str,
#         system_prompt: Optional[str] = None,
#         temperature: Optional[float] = None,
#         top_p: Optional[float] = None,
#         max_output_tokens: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         """
#         Convenience wrapper for single-turn generation.

#         Returns:
#         {
#             "content": str,
#             "usage": {
#                 "prompt_tokens": int,
#                 "completion_tokens": int,
#                 "total_tokens": int
#             },
#             "raw_response": dict
#         }
#         """
#         if not user_prompt:
#             raise ValueError("user_prompt must not be empty.")

#         final_system_prompt = system_prompt or self.default_system_prompt
#         messages = [
#             {"role": "system", "content": final_system_prompt},
#             {"role": "user", "content": user_prompt},
#         ]

#         return self.chat_with_usage(
#             messages=messages,
#             temperature=temperature,
#             top_p=top_p,
#             max_output_tokens=max_output_tokens,
#         )



import os
from typing import Optional, List, Dict, Any

from openai import OpenAI, OpenAIError


class QianfanAPIError(Exception):
    """Raised when Alibaba Bailian / DashScope API returns an error."""
    pass


class QianfanClient:
    """
    Alibaba Bailian / DashScope OpenAI-compatible chat client.

    Keep the old QianfanClient public interface unchanged:
    - generate() returns plain text
    - chat() returns plain text
    - generate_with_usage() returns content + usage + raw_response
    - chat_with_usage() returns content + usage + raw_response
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen2.5-7b-instruct",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
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

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _extract_content(self, completion: Any) -> str:
        try:
            content = completion.choices[0].message.content
        except Exception as exc:
            raw = self._to_raw_response(completion)
            raise QianfanAPIError(f"Unexpected Alibaba response format: {raw}") from exc

        if content is None:
            return ""
        return str(content)

    def _extract_usage(self, completion: Any) -> Dict[str, int]:
        usage = getattr(completion, "usage", None)

        if usage is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _to_raw_response(self, completion: Any) -> Dict[str, Any]:
        if hasattr(completion, "model_dump"):
            return completion.model_dump()
        if isinstance(completion, dict):
            return completion
        return {"raw": repr(completion)}

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        result = self.chat_with_usage(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        return result["content"]

    def chat_with_usage(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not messages:
            raise ValueError("messages must not be empty.")

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Important: keep non-streaming for exact and simple usage accounting.
            "stream": False,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        # Qianfan used max_output_tokens.
        # OpenAI-compatible Chat API uses max_tokens.
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens

        try:
            completion = self.client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise QianfanAPIError(f"Alibaba Bailian API error: {exc}") from exc
        except Exception as exc:
            raise QianfanAPIError(f"Unexpected Alibaba Bailian API error: {exc}") from exc

        return {
            "content": self._extract_content(completion),
            "usage": self._extract_usage(completion),
            "raw_response": self._to_raw_response(completion),
        }

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        result = self.generate_with_usage(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        return result["content"]

    def generate_with_usage(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not user_prompt:
            raise ValueError("user_prompt must not be empty.")

        final_system_prompt = system_prompt or self.default_system_prompt

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat_with_usage(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )